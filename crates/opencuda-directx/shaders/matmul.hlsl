// matmul コンピュートシェーダー(DirectX 12、DXIL)。
//
// `opencuda-vulkan::real::VulkanDevice::ensure_matmul_args`と同じ契約:
// 引数は a(m×k, 行優先) / b(k×n, 行優先) / c(m×n, 行優先) / m / k / n の6個。
// C = A × B (単純な3重ループ、タイリング等の最適化は行わない最小実装)。

RWStructuredBuffer<float> a : register(u0); // m x k
RWStructuredBuffer<float> b : register(u1); // k x n
RWStructuredBuffer<float> c : register(u2); // m x n

cbuffer Constants : register(b0) {
    uint m;
    uint k;
    uint n;
};

#define MATMUL_ROOT_SIGNATURE \
    "UAV(u0), UAV(u1), UAV(u2), RootConstants(num32BitConstants=3, b0)"

[RootSignature(MATMUL_ROOT_SIGNATURE)]
[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint row = dtid.y;
    uint col = dtid.x;
    if (row < m && col < n) {
        float sum = 0.0;
        for (uint i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}
