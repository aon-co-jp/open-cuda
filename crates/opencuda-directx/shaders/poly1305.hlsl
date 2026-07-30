// Poly1305認証タグ計算コンピュートシェーダー(DirectX 12、DXIL)。
//
// RS-LinkFusionのChaCha20-Poly1305 AEAD(`accel.rs`)を完成させるための、
// 前回HANDOFFで「実装難度が高く見送り」としていたPoly1305認証タグの
// GPU実装(ユーザー指示: 日英Web検索・GitHub調査の上で実装せよ、への対応)。
//
// **設計方針・裏取り**: Linuxカーネル/OpenSSL/H. Peter Anvinの実装ではなく
// Poly1305のリファレンス実装として広く使われる公開ドメイン実装
// "poly1305-donna"(Andrew Moon作、32bit版)のアルゴリズムを、日英Web検索
// (loup-vaillant.fr "The design of Poly1305"、Qiita/晴耕雨読の日本語解説
// 記事等)で二重に裏取りした上でそのまま踏襲する: 130ビットの数を
// 5個の26bit limb(r0..r4/h0..h4)で表現し、法 p=2^130-5 上での
// 加算・乗算を32bit整数演算のみで行う。
//
// **HLSL特有の制約への対応**: poly1305-donna-32はCの`unsigned long long`
// (64bit)型で中間積を保持するが、DXILのSM6.0でも64bit整数演算
// (`uint64_t`)はオプション機能(Int64ShaderOps)であり、GT730のような
// 旧世代GPUでの対応可否が不明(ChaCha20実装時と同じ「実機で本当に動くか
// 不明な機能には頼らない」方針)。そのため本シェーダは64bit整数型を
// 一切使わず、32bit×32bit→64bit(hi,lo)ペアの乗算(`umul32`)・64bit加算
// (`uadd64`)・64bit右シフト(`ushr64_lo`)を32bitのみのビット演算で自前
// 実装し、poly1305-donna-32の`unsigned long long`演算をすべてこれらの
// ヘルパーへ置き換える形で移植した。
//
// **スコープの正直な限定**: 1スレッドが1メッセージ全体を先頭から末尾まで
// 逐次処理する(Poly1305はh_new=(h_old+m_i)*r mod pという逐次依存の
// チェーンのため、1メッセージ内のブロック間では並列化できない——
// 並列化するにはr^kの冪乗事前計算+並列リダクションが必要だが、今回は
// スコープ外とし、代わりに「多数の独立した小さいメッセージ(RS-LinkFusion
// が扱うMTU程度のネットワークパケット)を1スレッド1メッセージで一括処理
// する」バッチ並列化とした——これはRS-LinkFusionの実際の利用形態
// (多数の独立パケット)によく合致する設計判断)。また、メッセージ長は
// 16バイトの整数倍のみ対応(Poly1305本来の「最後の不完全ブロックへの
// パディング」処理は未実装、呼び出し側が16バイト境界にパディング済みの
// データを渡す前提)。

RWStructuredBuffer<uint> data : register(u0);        // メッセージデータ(u32ワード、message_i は data[i*max_blocks*4 ..]
RWStructuredBuffer<uint> keys : register(u1);         // 1メッセージあたり8 u32(r生の16バイト+pad/sの16バイト)
RWStructuredBuffer<uint> block_counts : register(u2); // 1メッセージあたり1 uint(処理する16バイトブロック数)
RWStructuredBuffer<uint> tags : register(u3);         // 出力タグ、1メッセージあたり4 u32(16バイト)

cbuffer Constants : register(b0) {
    uint num_messages;
    uint max_blocks;
};

// 32bit×32bit→64bit(hi,lo)符号無し乗算。桁上げも含めて厳密。
void umul32(uint a, uint b, out uint hi, out uint lo) {
    uint a_lo = a & 0xFFFFu;
    uint a_hi = a >> 16u;
    uint b_lo = b & 0xFFFFu;
    uint b_hi = b >> 16u;

    uint p0 = a_lo * b_lo;
    uint p1 = a_lo * b_hi;
    uint p2 = a_hi * b_lo;
    uint p3 = a_hi * b_hi;

    uint mid = p1 + p2;
    uint mid_carry = (mid < p1) ? 1u : 0u;

    uint lo_result = p0 + (mid << 16u);
    uint lo_carry = (lo_result < p0) ? 1u : 0u;

    hi = p3 + (mid >> 16u) + (mid_carry << 16u) + lo_carry;
    lo = lo_result;
}

// (hi,lo) += (add_hi,add_lo)
void uadd64(inout uint hi, inout uint lo, uint add_hi, uint add_lo) {
    uint new_lo = lo + add_lo;
    uint carry = (new_lo < lo) ? 1u : 0u;
    lo = new_lo;
    hi = hi + add_hi + carry;
}

// (hi,lo) を n ビット(0<n<32)右シフトした結果の下位32bit。
// 呼び出し側は結果が32bitに収まることを保証する(poly1305-donnaの
// 桁上げ量の上限解析による、参照実装と同一の不変条件)。
uint ushr64_lo(uint hi, uint lo, uint n) {
    return (lo >> n) | (hi << (32u - n));
}

// a + b + carry_in(0 or 1) -> (result, carry_out(0 or 1))。32bit加算の
// 繰り上げ連鎖(mac = (h+pad) mod 2^128 の複数ワード加算)に使う。
void addc(uint a, uint b, uint carry_in, out uint result, out uint carry_out) {
    uint s = a + b;
    uint c1 = (s < a) ? 1u : 0u;
    uint s2 = s + carry_in;
    uint c2 = (s2 < s) ? 1u : 0u;
    result = s2;
    carry_out = c1 | c2;
}

#define POLY1305_ROOT_SIGNATURE \
    "UAV(u0), UAV(u1), UAV(u2), UAV(u3), RootConstants(num32BitConstants=2, b0)"

[RootSignature(POLY1305_ROOT_SIGNATURE)]
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint msg = dtid.x;
    if (msg >= num_messages) {
        return;
    }

    // --- r のクランプ(poly1305-donna-32のkey→r変換をそのまま踏襲) ---
    uint keyBase = msg * 8u;
    uint t0 = keys[keyBase + 0u];
    uint t1 = keys[keyBase + 1u];
    uint t2 = keys[keyBase + 2u];
    uint t3 = keys[keyBase + 3u];
    uint pad0 = keys[keyBase + 4u];
    uint pad1 = keys[keyBase + 5u];
    uint pad2 = keys[keyBase + 6u];
    uint pad3 = keys[keyBase + 7u];

    uint r0 = t0 & 0x3ffffffu;
    uint r1 = ((t0 >> 26u) | (t1 << 6u)) & 0x3ffff03u;
    uint r2 = ((t1 >> 20u) | (t2 << 12u)) & 0x3ffc0ffu;
    uint r3 = ((t2 >> 14u) | (t3 << 18u)) & 0x3f03fffu;
    uint r4 = (t3 >> 8u) & 0x00fffffu;

    uint s1 = r1 * 5u;
    uint s2 = r2 * 5u;
    uint s3 = r3 * 5u;
    uint s4 = r4 * 5u;

    uint h0 = 0u, h1 = 0u, h2 = 0u, h3 = 0u, h4 = 0u;

    uint blocks = block_counts[msg];
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }
    uint dataBase = msg * max_blocks * 4u;

    for (uint j = 0u; j < blocks; j++) {
        uint off = dataBase + j * 4u;
        uint m0 = data[off + 0u];
        uint m1 = data[off + 1u];
        uint m2 = data[off + 2u];
        uint m3 = data[off + 3u];

        // h += m (常に完全な16バイトブロックのみ扱う、hibit=1<<24固定)
        h0 += m0 & 0x3ffffffu;
        h1 += ((m0 >> 26u) | (m1 << 6u)) & 0x3ffffffu;
        h2 += ((m1 >> 20u) | (m2 << 12u)) & 0x3ffffffu;
        h3 += ((m2 >> 14u) | (m3 << 18u)) & 0x3ffffffu;
        h4 += (m3 >> 8u) | (1u << 24u);

        // h *= r (mod 2^130-5 は次の桁上げ処理で行う)
        uint d0hi, d0lo, d1hi, d1lo, d2hi, d2lo, d3hi, d3lo, d4hi, d4lo;
        uint phi, plo;

        d0hi = 0u; d0lo = 0u;
        umul32(h0, r0, phi, plo); uadd64(d0hi, d0lo, phi, plo);
        umul32(h1, s4, phi, plo); uadd64(d0hi, d0lo, phi, plo);
        umul32(h2, s3, phi, plo); uadd64(d0hi, d0lo, phi, plo);
        umul32(h3, s2, phi, plo); uadd64(d0hi, d0lo, phi, plo);
        umul32(h4, s1, phi, plo); uadd64(d0hi, d0lo, phi, plo);

        d1hi = 0u; d1lo = 0u;
        umul32(h0, r1, phi, plo); uadd64(d1hi, d1lo, phi, plo);
        umul32(h1, r0, phi, plo); uadd64(d1hi, d1lo, phi, plo);
        umul32(h2, s4, phi, plo); uadd64(d1hi, d1lo, phi, plo);
        umul32(h3, s3, phi, plo); uadd64(d1hi, d1lo, phi, plo);
        umul32(h4, s2, phi, plo); uadd64(d1hi, d1lo, phi, plo);

        d2hi = 0u; d2lo = 0u;
        umul32(h0, r2, phi, plo); uadd64(d2hi, d2lo, phi, plo);
        umul32(h1, r1, phi, plo); uadd64(d2hi, d2lo, phi, plo);
        umul32(h2, r0, phi, plo); uadd64(d2hi, d2lo, phi, plo);
        umul32(h3, s4, phi, plo); uadd64(d2hi, d2lo, phi, plo);
        umul32(h4, s3, phi, plo); uadd64(d2hi, d2lo, phi, plo);

        d3hi = 0u; d3lo = 0u;
        umul32(h0, r3, phi, plo); uadd64(d3hi, d3lo, phi, plo);
        umul32(h1, r2, phi, plo); uadd64(d3hi, d3lo, phi, plo);
        umul32(h2, r1, phi, plo); uadd64(d3hi, d3lo, phi, plo);
        umul32(h3, r0, phi, plo); uadd64(d3hi, d3lo, phi, plo);
        umul32(h4, s4, phi, plo); uadd64(d3hi, d3lo, phi, plo);

        d4hi = 0u; d4lo = 0u;
        umul32(h0, r4, phi, plo); uadd64(d4hi, d4lo, phi, plo);
        umul32(h1, r3, phi, plo); uadd64(d4hi, d4lo, phi, plo);
        umul32(h2, r2, phi, plo); uadd64(d4hi, d4lo, phi, plo);
        umul32(h3, r1, phi, plo); uadd64(d4hi, d4lo, phi, plo);
        umul32(h4, r0, phi, plo); uadd64(d4hi, d4lo, phi, plo);

        // (部分的な) h %= p、桁上げの伝播(poly1305-donna-32と同一の手順)
        uint c;
        c = ushr64_lo(d0hi, d0lo, 26u); h0 = d0lo & 0x3ffffffu;
        uadd64(d1hi, d1lo, 0u, c);      c = ushr64_lo(d1hi, d1lo, 26u); h1 = d1lo & 0x3ffffffu;
        uadd64(d2hi, d2lo, 0u, c);      c = ushr64_lo(d2hi, d2lo, 26u); h2 = d2lo & 0x3ffffffu;
        uadd64(d3hi, d3lo, 0u, c);      c = ushr64_lo(d3hi, d3lo, 26u); h3 = d3lo & 0x3ffffffu;
        uadd64(d4hi, d4lo, 0u, c);      c = ushr64_lo(d4hi, d4lo, 26u); h4 = d4lo & 0x3ffffffu;
        h0 += c * 5u;                   c = h0 >> 26u;                  h0 = h0 & 0x3ffffffu;
        h1 += c;
    }

    // --- 最終処理(poly1305-donna-32::poly1305_finishをそのまま踏襲) ---
    uint c;
    c = h1 >> 26u; h1 &= 0x3ffffffu;
    h2 += c; c = h2 >> 26u; h2 &= 0x3ffffffu;
    h3 += c; c = h3 >> 26u; h3 &= 0x3ffffffu;
    h4 += c; c = h4 >> 26u; h4 &= 0x3ffffffu;
    h0 += c * 5u; c = h0 >> 26u; h0 &= 0x3ffffffu;
    h1 += c;

    uint g0 = h0 + 5u; c = g0 >> 26u; g0 &= 0x3ffffffu;
    uint g1 = h1 + c; c = g1 >> 26u; g1 &= 0x3ffffffu;
    uint g2 = h2 + c; c = g2 >> 26u; g2 &= 0x3ffffffu;
    uint g3 = h3 + c; c = g3 >> 26u; g3 &= 0x3ffffffu;
    uint g4 = h4 + c - (1u << 26u);

    uint mask = (g4 >> 31u) - 1u; // g4の最上位ビットが1(=負、つまりh<p)ならmask=0、そうでなければ0xffffffff
    g0 &= mask; g1 &= mask; g2 &= mask; g3 &= mask; g4 &= mask;
    uint notMask = ~mask;
    h0 = (h0 & notMask) | g0;
    h1 = (h1 & notMask) | g1;
    h2 = (h2 & notMask) | g2;
    h3 = (h3 & notMask) | g3;

    // h = h mod 2^128 (26bit limb×4個 -> 32bit word×4個への詰め直し)
    uint out0 = (h0 | (h1 << 26u));
    uint out1 = ((h1 >> 6u) | (h2 << 20u));
    uint out2 = ((h2 >> 12u) | (h3 << 14u));
    uint out3 = ((h3 >> 18u) | (h4 << 8u));

    // mac = (h + pad) mod 2^128
    uint carry, r0out, r1out, r2out;
    addc(out0, pad0, 0u, r0out, carry);
    addc(out1, pad1, carry, r1out, carry);
    addc(out2, pad2, carry, r2out, carry);
    out0 = r0out; out1 = r1out; out2 = r2out;
    out3 = out3 + pad3 + carry;

    uint tagBase = msg * 4u;
    tags[tagBase + 0u] = out0;
    tags[tagBase + 1u] = out1;
    tags[tagBase + 2u] = out2;
    tags[tagBase + 3u] = out3;
}
