//! ソフトウェア四倍精度(F128)——**GPUハードウェア加速ではない**。
//!
//! 2026-09-03、ユーザー指示「32GB級カード(NVIDIA/AMD/Intel)を前提に
//! F16/F32/F64/F128まで見据えて開発」への対応。F16(`half::f16`)/F32/F64は
//! いずれもNVIDIA/AMD/Intelのいずれかの世代でGPUネイティブ演算がある一方、
//! **FP128(quad precision)をネイティブ実行できるGPUはNVIDIA/AMD/Intelの
//! どの製品にも存在しない**(コンシューマ・データセンター問わず、2026年
//! 時点で確認できる限り皆無)。Rust `std` にも安定版の`f128`プリミティブは
//! 無い(nightly限定の実験的機能のみ)。
//!
//! そのため本モジュールは、**double-double(倍々精度)** という古典的な
//! ソフトウェアエミュレーション手法(Dekker 1971 / Knuth `TAOCP` vol.2の
//! `two_sum`/`two_prod`アルゴリズム、QD/GNU MPFRの`dd_real`と同系統)で
//! 「2つのf64の組で約106ビット仮数部相当の精度」を提供する。
//!
//! **正直な開示(このモジュールの位置づけ)**:
//! - これは**CPU側のソフトウェア演算**であり、GPU上でのFP128命令は
//!   一切発行しない(発行できるGPUが存在しないため)。
//! - 目的は**性能ではなく数値的正確性**——Kahan和のような桁落ちに敏感な
//!   縮約(reduction)処理で、f64単体では失われる精度を確保するための道具。
//! - 四則演算(Add/Sub/Mul/Div)のみ実装し、超越関数(sqrt/exp/log等)は
//!   未実装(必要になった時点で追加する)。
//! - 相対誤差は概ね2^-106程度(f64の2^-52よりはるかに高精度)だが、
//!   `qd`/`twofloat`のような専用crateほど枯れた実装ではない
//!   (自己完結・追加のネットワーク依存を避けるための自前実装、テストで
//!   往復精度をf64単体と比較検証している)。

/// Double-double(倍々精度)浮動小数点数。`hi + lo` が実際の値を表し、
/// `|lo| <= 0.5 ulp(hi)` となるよう正規化して保つ(Dekkerの不変条件)。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DoubleDouble {
    pub hi: f64,
    pub lo: f64,
}

impl DoubleDouble {
    pub const ZERO: DoubleDouble = DoubleDouble { hi: 0.0, lo: 0.0 };

    #[inline]
    pub fn from_f64(v: f64) -> Self {
        DoubleDouble { hi: v, lo: 0.0 }
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    /// Knuth `two_sum`: `a+b` を丸め誤差ぶんまで含めて厳密に `(s, e)` へ分解
    /// する(`s+e == a+b` が浮動小数点演算の意味で厳密に成り立つ)。
    #[inline]
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        let err = (a - (s - bb)) + (b - bb);
        (s, err)
    }

    /// `two_sum`の高速版(`|a| >= |b|`が既知の場合、Dekkerの`quick_two_sum`)。
    #[inline]
    fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let err = b - (s - a);
        (s, err)
    }

    /// Dekker `two_prod`: `a*b` を `(p, e)` へ厳密分解。
    /// FMA(`f64::mul_add`)が使える場合は1回のFMAで誤差項が厳密に求まる
    /// (split-based Veltkamp分解より単純・高速、Muller et al.
    /// "Handbook of Floating-Point Arithmetic" 記載の標準手法)。
    #[inline]
    fn two_prod(a: f64, b: f64) -> (f64, f64) {
        let p = a * b;
        let e = a.mul_add(b, -p);
        (p, e)
    }

    /// `self + other`(double-double同士の加算、Dekkerのアルゴリズム)。
    /// `std::ops::Add`実装(下記)から呼ばれる本体。
    fn add_impl(self, other: DoubleDouble) -> DoubleDouble {
        let (s, e) = Self::two_sum(self.hi, other.hi);
        let e = e + self.lo + other.lo;
        let (hi, lo) = Self::quick_two_sum(s, e);
        DoubleDouble { hi, lo }
    }

    fn sub_impl(self, other: DoubleDouble) -> DoubleDouble {
        self.add_impl(other.neg_impl())
    }

    fn neg_impl(self) -> DoubleDouble {
        DoubleDouble { hi: -self.hi, lo: -self.lo }
    }

    /// `self * other`。
    fn mul_impl(self, other: DoubleDouble) -> DoubleDouble {
        let (p, e) = Self::two_prod(self.hi, other.hi);
        let e = e + self.hi * other.lo + self.lo * other.hi;
        let (hi, lo) = Self::quick_two_sum(p, e);
        DoubleDouble { hi, lo }
    }

    /// `self / other`(Newton補正1回、`qd`ライブラリと同じ標準手法)。
    fn div_impl(self, other: DoubleDouble) -> DoubleDouble {
        let q1 = self.hi / other.hi;
        let r = self.sub_impl(other.mul_impl(DoubleDouble::from_f64(q1)));
        let q2 = r.to_f64() / other.hi;
        let (hi, lo) = Self::quick_two_sum(q1, q2);
        DoubleDouble { hi, lo }
    }
}

impl std::ops::Add for DoubleDouble {
    type Output = DoubleDouble;
    fn add(self, rhs: DoubleDouble) -> DoubleDouble {
        self.add_impl(rhs)
    }
}
impl std::ops::Sub for DoubleDouble {
    type Output = DoubleDouble;
    fn sub(self, rhs: DoubleDouble) -> DoubleDouble {
        self.sub_impl(rhs)
    }
}
impl std::ops::Neg for DoubleDouble {
    type Output = DoubleDouble;
    fn neg(self) -> DoubleDouble {
        self.neg_impl()
    }
}
impl std::ops::Mul for DoubleDouble {
    type Output = DoubleDouble;
    fn mul(self, rhs: DoubleDouble) -> DoubleDouble {
        self.mul_impl(rhs)
    }
}
impl std::ops::Div for DoubleDouble {
    type Output = DoubleDouble;
    fn div(self, rhs: DoubleDouble) -> DoubleDouble {
        self.div_impl(rhs)
    }
}
impl std::iter::Sum for DoubleDouble {
    fn sum<I: Iterator<Item = DoubleDouble>>(iter: I) -> Self {
        iter.fold(DoubleDouble::ZERO, |a, b| a + b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_f64_to_f64_round_trips_exactly() {
        for v in [0.0, 1.0, -1.0, 2.71828182845901, 1e300, 1e-300] {
            assert_eq!(DoubleDouble::from_f64(v).to_f64(), v);
        }
    }

    #[test]
    fn add_matches_f64_for_well_conditioned_values() {
        let a = DoubleDouble::from_f64(1.5);
        let b = DoubleDouble::from_f64(2.25);
        assert_eq!((a + b).to_f64(), 3.75);
    }

    #[test]
    fn mul_and_div_are_consistent_round_trip() {
        let a = DoubleDouble::from_f64(1.23456789);
        let b = DoubleDouble::from_f64(9.87654321);
        let prod = a * b;
        let back = prod / b;
        // 往復誤差はf64のulpのごく僅かなオーダーに収まるべき。
        assert!((back.to_f64() - a.to_f64()).abs() < 1e-12);
    }

    /// double-doubleが実際にf64単体より高精度であることを示す本命テスト。
    /// 病的に条件の悪い和(巨大な値+多数の微小な値)を、f64累積・
    /// double-double累積の両方で計算し、真値(有理数的に厳密な値)との
    /// 誤差を比較する。
    #[test]
    fn dd_summation_is_more_accurate_than_plain_f64_for_ill_conditioned_sum() {
        // 1e16(f64の精度限界付近の大きさ)に、1.0を1000万回足す。
        // 真の合計は 1e16 + 1e7 = 10_000_000_010_000_000.0 のはず。
        let big = 1.0e16_f64;
        let n = 10_000_000_i64;

        let mut f64_acc = big;
        for _ in 0..n {
            f64_acc += 1.0;
        }

        let mut dd_acc = DoubleDouble::from_f64(big);
        let one = DoubleDouble::from_f64(1.0);
        for _ in 0..n {
            dd_acc = dd_acc + one;
        }

        let true_value = big + n as f64; // f64として表現可能な厳密値
        let f64_err = (f64_acc - true_value).abs();
        let dd_err = (dd_acc.to_f64() - true_value).abs();

        // f64単体は桁落ちで真値から大きく外れる一方、double-doubleは
        // 真値と厳密に一致するはず。
        assert!(dd_err < f64_err, "dd_err={dd_err} should be < f64_err={f64_err}");
        assert_eq!(dd_acc.to_f64(), true_value, "double-double should be exact for this case");
    }

    #[test]
    fn neg_and_sub_are_consistent() {
        let a = DoubleDouble::from_f64(5.0);
        let b = DoubleDouble::from_f64(3.0);
        assert_eq!((a - b).to_f64(), 2.0);
        assert_eq!((b - a).to_f64(), -2.0);
    }
}
