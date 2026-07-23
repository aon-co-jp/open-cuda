// vector_add コンピュートシェーダー(DirectX 12、DXIL)。
//
// ルートシグネチャをHLSL内へ直接埋め込む([RootSignature(...)]属性)。
// これによりdxcがコンパイルしたDXILバイト列自体にルートシグネチャが
// 同梱され、Rust側は`ID3D12Device::CreateRootSignature`へこのバイト列を
// そのまま渡すだけでよい(別途C++/Rust側でルートシグネチャを組み立てる
// 必要がない)。ディスクリプタヒープも使わず、3つのUAVバッファは
// ルートディスクリプタとして直接バインドする(`SetComputeRootUnorderedAccessView`)。

RWStructuredBuffer<float> a : register(u0);
RWStructuredBuffer<float> b : register(u1);
RWStructuredBuffer<float> c : register(u2);

cbuffer Constants : register(b0) {
    uint n;
};

#define VECTOR_ADD_ROOT_SIGNATURE \
    "UAV(u0), UAV(u1), UAV(u2), RootConstants(num32BitConstants=1, b0)"

[RootSignature(VECTOR_ADD_ROOT_SIGNATURE)]
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint i = dtid.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
