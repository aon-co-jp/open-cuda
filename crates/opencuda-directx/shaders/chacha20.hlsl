// ChaCha20ストリーム暗号コンピュートシェーダー(DirectX 12、DXIL)。
//
// RFC 8439のブロック関数をそのまま実装(20ラウンド=column round×4+
// diagonal round×4を10セット)。1スレッドが64バイト(16ワード)の
// ブロック1個を担当し、生成したキーストリームをバッファへXORする
// (in-place、暗号化と復号は同じ演算——ストリーム暗号の性質)。
//
// **正直な開示**: これはRS-LinkFusion(`accel.rs`のChaCha20Poly1305)が
// 実際に使うAEAD全体(ChaCha20-Poly1305)のうち、認証タグ計算
// (Poly1305)を含まないChaCha20暗号化部分のみのGPU実装デモンストレー
// ションである。RFC 8439 §2.3.2の公式テストベクタで数値一致を検証
// 済みだが、本番のAEAD実装として`accel.rs`へ組み込むには別途
// Poly1305認証タグの実装、およびRS-LinkFusion側での実際のペイロード
// サイズ(MTU程度、数百〜数千バイト)でのH2D/D2Hオーバーヘッドが
// CPU実装(`chacha20poly1305`クレート)に対して実利益を生むかの
// ベンチマークが必要(open-cuda CLAUDE.md HANDOFF参照)。

RWStructuredBuffer<uint> data : register(u0); // 暗号化/復号対象、u32ワード列(リトルエンディアン)

// **正直な開示・実バグの修正(2026-07-23)**: 当初 `uint key[8]`/
// `uint nonce[3]` という配列宣言だったが、HLSLのcbufferパッキング規則
// では**スカラー配列の各要素が16バイト境界にパディングされる**
// (`float weights[3]`が3*16=48バイトを占める、というよく知られた罠と
// 同じ)。Rust側は`SetComputeRoot32BitConstant`で13個のdwordを
// 隙間なく詰めて渡す設計のため、配列パディングが入るとHLSL側が読む
// バイトオフセットとRust側が書き込むオフセットがズレ、
// key/nonce/counter_base/length_wordsすべてが実際には無関係な値
// (実質ゼロに近い値)を読んでしまい、キーストリームが実質0になって
// 平文がそのまま出力される、という実バグを引き起こしていた(実機
// テストで発覚、`cargo test`が値の不一致で検出)。個別スカラー
// フィールドに書き換えることでパディング無しの密なレイアウトにし、
// Rust側の詰め込み方と一致させる。
cbuffer Constants : register(b0) {
    uint key0; uint key1; uint key2; uint key3;
    uint key4; uint key5; uint key6; uint key7;
    uint nonce0; uint nonce1; uint nonce2;
    uint counter_base; // ブロックカウンタの開始値
    uint length_words;  // dataバッファの総ワード数
};

uint rotl(uint x, uint n) {
    return (x << n) | (x >> (32u - n));
}

void quarterRound(inout uint a, inout uint b, inout uint c, inout uint d) {
    a += b; d ^= a; d = rotl(d, 16u);
    c += d; b ^= c; b = rotl(b, 12u);
    a += b; d ^= a; d = rotl(d, 8u);
    c += d; b ^= c; b = rotl(b, 7u);
}

#define CHACHA20_ROOT_SIGNATURE \
    "UAV(u0), RootConstants(num32BitConstants=13, b0)"

[RootSignature(CHACHA20_ROOT_SIGNATURE)]
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint blockIndex = dtid.x;
    uint wordOffset = blockIndex * 16u;
    if (wordOffset >= length_words) {
        return;
    }

    uint state[16];
    state[0] = 0x61707865u;
    state[1] = 0x3320646eu;
    state[2] = 0x79622d32u;
    state[3] = 0x6b206574u;
    state[4] = key0;
    state[5] = key1;
    state[6] = key2;
    state[7] = key3;
    state[8] = key4;
    state[9] = key5;
    state[10] = key6;
    state[11] = key7;
    state[12] = counter_base + blockIndex;
    state[13] = nonce0;
    state[14] = nonce1;
    state[15] = nonce2;

    uint working[16];
    for (uint i = 0; i < 16u; i++) {
        working[i] = state[i];
    }

    for (uint round = 0; round < 10u; round++) {
        quarterRound(working[0], working[4], working[8], working[12]);
        quarterRound(working[1], working[5], working[9], working[13]);
        quarterRound(working[2], working[6], working[10], working[14]);
        quarterRound(working[3], working[7], working[11], working[15]);

        quarterRound(working[0], working[5], working[10], working[15]);
        quarterRound(working[1], working[6], working[11], working[12]);
        quarterRound(working[2], working[7], working[8], working[13]);
        quarterRound(working[3], working[4], working[9], working[14]);
    }

    for (uint j = 0; j < 16u; j++) {
        uint idx = wordOffset + j;
        if (idx < length_words) {
            uint keystream = working[j] + state[j];
            data[idx] = data[idx] ^ keystream;
        }
    }
}
