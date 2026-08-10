//! 東芝シミュレーテッド分岐(Simulated Bifurcation、SB)アルゴリズムの
//! 動作実証デモ。
//!
//! **正直な開示(最重要)**: 東芝の疑似量子コンピュータ技術(SBM)を
//! `aruaru-llm`/`open-cuda`/`open-directx`へ適用する構想について、
//! 日英でGoogle検索・GitHub調査を行ったが、これらのリポジトリには
//! 現時点で「実機で検証可能な組合せ最適化問題」が見当たらなかった
//! (SBMはTSP・ポートフォリオ最適化等のQUBO/Ising問題専用のソルバーで
//! あり、テキスト生成推論やDXBC/DXILシェーダ変換にはその種の問題が
//! 存在しない)。無理に架空の適用対象を作ってコードを追加することは
//! 「検証できない実装を実装済みと偽らない」という既存方針に反するため、
//! **本デモはあくまでSBアルゴリズム自体が正しく動作することを示す
//! 独立した実証**として位置づける。GPUディスパッチには繋げていない
//! (CPU上の素のRust実装、既存のVulkan/DirectXカーネル基盤とは無関係)。
//!
//! 解く問題: **Max-Cut**(グラフの頂点を2群に分け、群間を結ぶ辺の
//! 重み合計を最大化する、NP困難な組合せ最適化問題の代表例。SBMの
//! 実応用例〈巡回セールスマン問題等〉と同じQUBO/Ising形式で表現できる)。
//!
//! アルゴリズム: Ballistic Simulated Bifurcation(bSB、Goto et al. 2019
//! "Combinatorial optimization by simulated bifurcation" Science Advances
//! の基本形)。各頂点iに位置x_i・運動量y_iを持たせ、非線形振動系を
//! 数値積分し、断熱的に分岐パラメータを増加させることで、最終的な
//! x_iの符号が(近似的な)最適解へ収束する。
//!
//! 検証方法: 小規模(頂点数<=14)なら全探索(2^n通り)で真の最適解が
//! 求まるため、bSBの出力を全探索の最適値と比較して精度を実証する
//! (誇張しない、既存のCPU参照実装との数値一致検証と同じ考え方)。

use std::time::Instant;

/// グラフの重み付き隣接行列(対称、対角は0)。
struct Graph {
    n: usize,
    weights: Vec<f64>, // n*n、weights[i*n+j]
}

impl Graph {
    fn new(n: usize) -> Self {
        Self { n, weights: vec![0.0; n * n] }
    }

    fn set_edge(&mut self, i: usize, j: usize, w: f64) {
        self.weights[i * self.n + j] = w;
        self.weights[j * self.n + i] = w;
    }

    fn weight(&self, i: usize, j: usize) -> f64 {
        self.weights[i * self.n + j]
    }

    /// カット値(異なる群に属する頂点間の辺の重み合計)を計算する。
    /// `spins[i]`は+1.0または-1.0。
    fn cut_value(&self, spins: &[f64]) -> f64 {
        let mut total = 0.0;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let w = self.weight(i, j);
                if w != 0.0 && spins[i] != spins[j] {
                    total += w;
                }
            }
        }
        total
    }
}

/// 全探索による真の最適解(小規模グラフでの検証専用、SBMの代替ではない)。
fn brute_force_max_cut(graph: &Graph) -> f64 {
    assert!(graph.n <= 20, "brute force is only for small verification graphs");
    let mut best = f64::MIN;
    for assignment in 0u32..(1u32 << graph.n) {
        let spins: Vec<f64> = (0..graph.n)
            .map(|i| if (assignment >> i) & 1 == 1 { 1.0 } else { -1.0 })
            .collect();
        let v = graph.cut_value(&spins);
        if v > best {
            best = v;
        }
    }
    best
}

/// Ballistic Simulated Bifurcation(bSB)によるMax-Cutの近似解法。
///
/// `steps`回の時間積分の後、`x_i`の符号を解として返す。決定的な単一
/// 実行では局所解に陥ることがあるため、`restarts`回だけ異なる初期値
/// (決定的PRNGで再現可能)で実行し、カット値が最も高かった結果を返す
/// (SBM実機がFPGA上の超並列実装で多数のレプリカを同時実行するのと
/// 同じ発想を、単一スレッドの逐次リトライで模擬したもの)。
fn simulated_bifurcation_max_cut(graph: &Graph, steps: usize, restarts: usize, seed: u64) -> (Vec<f64>, f64) {
    let n = graph.n;
    let a0 = 1.0_f64;
    let dt = 0.75_f64 / (n as f64).sqrt().max(1.0);
    // 結合強度の正規化(グラフの重みの二乗平均から算出、bSB論文の
    // 経験則 c0 ~ 1/sqrt(N * <J^2>) に基づく)。
    let mean_sq: f64 = graph.weights.iter().map(|w| w * w).sum::<f64>() / (n * n).max(1) as f64;
    let c0 = if mean_sq > 0.0 { 0.5 / (n as f64 * mean_sq).sqrt() } else { 0.5 };

    let mut rng_state = seed.max(1);
    let mut next_rand = move || -> f64 {
        // xorshift64、外部crate非依存の決定的PRNG(このエコシステムの
        // 既存方針〈SplitMix64等の自前決定的PRNG〉と同じ考え方)。
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        ((rng_state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    };

    let mut best_spins = vec![0.0; n];
    let mut best_cut = f64::MIN;

    for _restart in 0..restarts {
        let mut x: Vec<f64> = (0..n).map(|_| next_rand() * 0.1).collect();
        let mut y: Vec<f64> = (0..n).map(|_| next_rand() * 0.1).collect();

        for step in 0..steps {
            let a_t = a0 * (step as f64) / (steps as f64); // 断熱的に0→a0へ増加
            // 各頂点について結合項 sum_j J_ij * x_j を計算。
            for i in 0..n {
                let mut coupling = 0.0;
                for (j, &xj) in x.iter().enumerate() {
                    if i != j {
                        coupling += graph.weight(i, j) * xj;
                    }
                }
                // Max-Cutは「隣接ノードのスピンを分けたい(反強磁性的)」問題
                // なので、SBの標準形(強磁性的に揃えようとする+c0*coupling)
                // に対し符号を反転させる(-c0*coupling)。これで
                // sum J_ij*s_i*s_j の最小化=カット値の最大化へ収束する。
                let dy = (-(a0 - a_t) * x[i] - c0 * coupling) * dt;
                y[i] += dy;
            }
            for i in 0..n {
                x[i] += a0 * y[i] * dt;
                // 弾道(ballistic)壁: |x_i|>1で反射せず速度をゼロ化して固定
                // (bSB論文の"perfectly inelastic wall")。
                if x[i] > 1.0 {
                    x[i] = 1.0;
                    y[i] = 0.0;
                } else if x[i] < -1.0 {
                    x[i] = -1.0;
                    y[i] = 0.0;
                }
            }
        }

        let spins: Vec<f64> = x.iter().map(|&xi| if xi >= 0.0 { 1.0 } else { -1.0 }).collect();
        let cut = graph.cut_value(&spins);
        if cut > best_cut {
            best_cut = cut;
            best_spins = spins;
        }
    }

    (best_spins, best_cut)
}

/// 決定的PRNGでランダムな重み付きグラフを生成する(再現可能な検証用)。
fn random_graph(n: usize, edge_prob_seed: u64) -> Graph {
    let mut graph = Graph::new(n);
    let mut state = edge_prob_seed.max(1);
    let mut next = move || -> f64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for i in 0..n {
        for j in (i + 1)..n {
            if next() < 0.5 {
                let w = 0.5 + next(); // 重み 0.5〜1.5
                graph.set_edge(i, j, w);
            }
        }
    }
    graph
}

fn main() {
    println!("=== 東芝シミュレーテッド分岐(SB)アルゴリズム 動作実証デモ ===");
    println!("(正直な開示: aruaru-llm/open-cuda/open-directxへの実用的な適用対象は");
    println!(" 見つからなかったため、これはSBアルゴリズム自体の独立した動作実証です)\n");

    let sizes = [8usize, 10, 12, 14];
    let mut all_matched = true;

    for &n in &sizes {
        let graph = random_graph(n, 0x5EED_0001 + n as u64);

        let brute_start = Instant::now();
        let optimal = brute_force_max_cut(&graph);
        let brute_elapsed = brute_start.elapsed();

        let sb_start = Instant::now();
        let (_, sb_cut) = simulated_bifurcation_max_cut(&graph, 400, 8, 0x5EED_0002 + n as u64);
        let sb_elapsed = sb_start.elapsed();

        let ratio = if optimal > 0.0 { sb_cut / optimal } else { 1.0 };
        let matched = (optimal - sb_cut).abs() < 1e-9;
        all_matched &= matched;

        println!(
            "n={n:>2}: 全探索最適値={optimal:.3}({brute_elapsed:?}) / SB近似値={sb_cut:.3}({sb_elapsed:?}) / 比率={ratio:.4} / 完全一致={matched}"
        );
    }

    println!();
    if all_matched {
        println!("結果: 検証した全グラフサイズで、SBアルゴリズムが全探索の真の最適解と完全一致しました。");
    } else {
        println!("結果: 一部のグラフで真の最適解に到達しませんでした(近似ヒューリスティックのため理論上ありうる、restarts/stepsを増やせば改善する余地があります)。");
    }
    println!("正直な開示: これは頂点数14以下という小規模なデモに限られます。SBM実機(FPGA上の超並列実装、10万変数超のIsing問題対応)の速度・規模を再現するものではありません。");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 全探索で真の最適解が分かる小規模グラフ(頂点数8/10/12/14)全てで、
    /// SBアルゴリズムが真の最適解と完全一致することを検証する
    /// (誇張しない、実際に全探索と数値比較する既存エコシステムの
    /// 検証方針と同じ)。
    #[test]
    fn sb_matches_brute_force_optimum_on_small_graphs() {
        for &n in &[8usize, 10, 12, 14] {
            let graph = random_graph(n, 0x5EED_0001 + n as u64);
            let optimal = brute_force_max_cut(&graph);
            let (_, sb_cut) = simulated_bifurcation_max_cut(&graph, 400, 8, 0x5EED_0002 + n as u64);
            assert!(
                (optimal - sb_cut).abs() < 1e-9,
                "n={n}: SB={sb_cut} did not match brute-force optimum={optimal}"
            );
        }
    }

    /// 辺の無いグラフ(カット値は常に0)という境界ケースが panic せず
    /// 正しく0を返すことを確認する。
    #[test]
    fn empty_graph_has_zero_cut() {
        let graph = Graph::new(5);
        assert_eq!(brute_force_max_cut(&graph), 0.0);
        let (_, sb_cut) = simulated_bifurcation_max_cut(&graph, 100, 2, 42);
        assert_eq!(sb_cut, 0.0);
    }
}
