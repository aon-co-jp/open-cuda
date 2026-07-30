//! # opencuda-directx
//!
//! DirectX 12 Compute バックエンド(Windows専用、`opencuda-vulkan`と並存する
//! オプトインバックエンド)。
//!
//! ## 正直な開示・経緯(2026-07-23)
//!
//! ユーザーから当初「open-cudaはDirectXのプラグインとして開発中」との
//! 認識が示されたが、実際には`opencuda-vulkan`(Vulkan Compute)が最初から
//! 唯一のGPUバックエンドだった。日英Web検索での裏取りの結果、
//! DXVK/vkd3d-protonのような実例はいずれも「DirectX(Windows専用API)→
//! Vulkan(クロスプラットフォームAPI)」という変換方向であり、逆方向
//! (DirectXを他OSへネイティブ移植)の実例は見つからなかった——つまり
//! クロスプラットフォーム対応という目標に対しては`opencuda-vulkan`の
//! 既存方針の方が近道である、という技術的判断をユーザーへ報告した。
//! その上でユーザーは「Vulkanは残しつつ、Windows向けに別途DirectX
//! バックエンドを追加する」(両方維持)という方針を選択した。本クレートは
//! その決定に基づく、**Windows専用のオプトイン追加バックエンド**である。
//!
//! ## 実装フェーズ(`opencuda-vulkan`のPhase 1.5パターンを踏襲)
//!
//! - **Phase 1(本クレート、実装済み)**: `DirectXMockDevice`(GPU/DirectX
//!   ランタイムなしでDXIL経路の契約を検証するシミュレータ)+
//!   `real-dx12` feature配下の実`DirectXDevice`(実際の`D3D12CreateDevice`
//!   でのデバイス列挙・コマンドキュー作成・UPLOAD/READBACKヒープ経由の
//!   実メモリ確保・h2d/d2h/d2dコピーまでを実機検証)。
//! - **Phase 2(未着手、正直な開示)**: `launch_kernel`の実際のDXIL
//!   ディスパッチ(ルートシグネチャ・Compute PSO・ディスクリプタヒープ・
//!   コマンドリスト記録・`ExecuteCommandLists`)。本クレートの
//!   `DirectXDevice::launch_kernel`は現時点では`GpuError::UnsupportedKernel`
//!   を返す——Vulkanバックエンドの`dispatch_spirv`に相当する処理が
//!   未実装であることを偽らない。

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail};
use opencuda_core::{
    CompiledKernel, DeviceInfo, DevicePtr, GpuDevice, GpuError, GpuVendor, KernelArg, KernelSource,
    LaunchConfig, Result,
};

/// DXBCコンテナ形式のマジックバイト(`b"DXBC"`、リトルエンディアンで先頭4バイト)。
/// DXILもDXBCコンテナに格納されるため、実DXILバイト列はこの先頭を持つ。
const DXBC_MAGIC: [u8; 4] = *b"DXBC";

#[derive(Default)]
struct Allocation {
    bytes: Vec<u8>,
}

/// GPU/DirectXランタイムなしでDXIL経路をテストするための代替デバイス。
///
/// `opencuda-vulkan::VulkanMockDevice`と同じ設計判断: 実DirectX 12
/// バックエンドではない。実装済み範囲を意図的に`vector_add`系のみに
/// 限定し、誇大に見えないようにする。
pub struct DirectXMockDevice {
    info: DeviceInfo,
    allocations: Mutex<HashMap<u64, Allocation>>,
    next_handle: AtomicU64,
}

impl DirectXMockDevice {
    pub fn new(id: usize) -> Arc<Self> {
        Arc::new(Self {
            info: DeviceInfo {
                id,
                vendor: GpuVendor::Unknown,
                name: "OpenCUDA DirectX Mock Device (DXIL path simulator, no GPU)".to_string(),
                total_memory: 512 * 1024 * 1024,
                compute_units: 1,
            },
            allocations: Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
        })
    }

    fn check_ptr(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.device_id as usize != self.info.id {
            return Err(GpuError::InvalidPtr(ptr).into());
        }
        if !self.allocations.lock().unwrap().contains_key(&ptr.addr) {
            return Err(GpuError::InvalidPtr(ptr).into());
        }
        Ok(())
    }

    fn validate_dxil(bytes: &[u8]) -> Result<()> {
        if bytes.len() < 4 {
            bail!("invalid DXIL: buffer is shorter than 4 bytes");
        }
        if bytes[..4] != DXBC_MAGIC {
            bail!("invalid DXIL: missing DXBC container magic \"DXBC\"");
        }
        Ok(())
    }

    fn read_f32_vec(&self, ptr: DevicePtr, n: usize) -> Result<Vec<f32>> {
        self.check_ptr(ptr)?;
        let map = self.allocations.lock().unwrap();
        let alloc = map.get(&ptr.addr).unwrap();
        let bytes = n
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow!("byte size overflow"))?;
        if bytes > alloc.bytes.len() {
            return Err(GpuError::InvalidPtr(ptr).into());
        }
        let mut out = Vec::with_capacity(n);
        for chunk in alloc.bytes[..bytes].chunks_exact(4) {
            out.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    }

    fn write_f32_vec(&self, ptr: DevicePtr, values: &[f32]) -> Result<()> {
        self.check_ptr(ptr)?;
        let mut map = self.allocations.lock().unwrap();
        let alloc = map.get_mut(&ptr.addr).unwrap();
        let bytes = values
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow!("byte size overflow"))?;
        if bytes > alloc.bytes.len() {
            return Err(GpuError::InvalidPtr(ptr).into());
        }
        for (chunk, value) in alloc.bytes[..bytes].chunks_exact_mut(4).zip(values.iter()) {
            chunk.copy_from_slice(&value.to_ne_bytes());
        }
        Ok(())
    }

    fn run_vector_add_simulation(&self, args: &[KernelArg]) -> Result<()> {
        if args.len() != 4 {
            bail!("vector_add expects 4 args: a, b, c, n");
        }
        let a = args[0].as_ptr().ok_or_else(|| anyhow!("arg0 must be pointer"))?;
        let b = args[1].as_ptr().ok_or_else(|| anyhow!("arg1 must be pointer"))?;
        let c = args[2].as_ptr().ok_or_else(|| anyhow!("arg2 must be pointer"))?;
        let n = args[3].as_usize().ok_or_else(|| anyhow!("arg3 must be usize"))?;

        let av = self.read_f32_vec(a, n)?;
        let bv = self.read_f32_vec(b, n)?;
        let mut cv = Vec::with_capacity(n);
        for i in 0..n {
            cv.push(av[i] + bv[i]);
        }
        self.write_f32_vec(c, &cv)
    }
}

impl GpuDevice for DirectXMockDevice {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        if bytes == 0 {
            return Err(GpuError::OutOfMemory(0).into());
        }
        let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
        self.allocations
            .lock()
            .unwrap()
            .insert(handle, Allocation { bytes: vec![0; bytes] });
        Ok(DevicePtr::new(handle, self.info.id as u32))
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        self.check_ptr(ptr)?;
        self.allocations.lock().unwrap().remove(&ptr.addr);
        Ok(())
    }

    fn memcpy_h2d(&self, dst: DevicePtr, src: &[u8]) -> Result<()> {
        self.check_ptr(dst)?;
        let mut map = self.allocations.lock().unwrap();
        let alloc = map.get_mut(&dst.addr).unwrap();
        if src.len() > alloc.bytes.len() {
            return Err(GpuError::OutOfMemory(src.len()).into());
        }
        alloc.bytes[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn memcpy_d2h(&self, dst: &mut [u8], src: DevicePtr) -> Result<()> {
        self.check_ptr(src)?;
        let map = self.allocations.lock().unwrap();
        let alloc = map.get(&src.addr).unwrap();
        if dst.len() > alloc.bytes.len() {
            return Err(GpuError::InvalidPtr(src).into());
        }
        dst.copy_from_slice(&alloc.bytes[..dst.len()]);
        Ok(())
    }

    fn memcpy_d2d(&self, dst: DevicePtr, src: DevicePtr, bytes: usize) -> Result<()> {
        self.check_ptr(dst)?;
        self.check_ptr(src)?;
        let mut map = self.allocations.lock().unwrap();
        let tmp = {
            let s = map.get(&src.addr).unwrap();
            if bytes > s.bytes.len() {
                return Err(GpuError::InvalidPtr(src).into());
            }
            s.bytes[..bytes].to_vec()
        };
        let d = map.get_mut(&dst.addr).unwrap();
        if bytes > d.bytes.len() {
            return Err(GpuError::InvalidPtr(dst).into());
        }
        d.bytes[..bytes].copy_from_slice(&tmp);
        Ok(())
    }

    fn launch_kernel(&self, kernel: &CompiledKernel, _cfg: &LaunchConfig, args: &[KernelArg]) -> Result<()> {
        match &kernel.source {
            KernelSource::Dxil(bytes) => Self::validate_dxil(bytes)?,
            other => return Err(GpuError::UnsupportedKernel(other.kind()).into()),
        }

        match kernel.name.as_str() {
            "vector_add" | "vector_add_f32" => self.run_vector_add_simulation(args),
            other => bail!("DirectXMockDevice only simulates vector_add/vector_add_f32; got kernel `{other}`"),
        }
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn supports_dxil(&self) -> bool {
        true
    }
}

pub fn enumerate(start_id: usize) -> Vec<Arc<dyn GpuDevice>> {
    vec![DirectXMockDevice::new(start_id)]
}

#[cfg(all(windows, feature = "real-dx12"))]
pub mod real;

#[cfg(all(windows, feature = "real-dx12"))]
pub use real::{enumerate_real, DirectXDevice, DirectXDiagnostics};

#[cfg(all(windows, feature = "real-dx12", test))]
mod real_hardware_tests {
    use super::real::DirectXDevice;
    use opencuda_core::alloc_buffer;

    /// 実機D3D12テスト。DirectX 12対応GPU/ドライバが無い環境では
    /// `eprintln!`でスキップする(Vulkanバックエンドの
    /// `sgemm_vulkan_generic_matches_cpu_naive_on_real_hardware`と同じ
    /// 「未検証をgreenに偽装しない」パターン)。
    #[test]
    fn real_d3d12_device_roundtrips_h2d_and_d2h_on_real_hardware() {
        let device = match DirectXDevice::new(0) {
            Ok(dev) => dev,
            Err(e) => {
                eprintln!("skipping real D3D12 test: {e}");
                return;
            }
        };

        let dev: std::sync::Arc<dyn opencuda_core::GpuDevice> = device;
        let data = b"real DirectX 12 upload-heap roundtrip test payload";
        let buf = alloc_buffer(&dev, data.len()).unwrap();
        buf.copy_from_host(data).unwrap();

        let mut out = vec![0u8; data.len()];
        buf.copy_to_host(&mut out).unwrap();
        assert_eq!(&out, data);
    }

    /// DXGIアダプタ列挙によるベンダー判定(2026-07-23追加)。実機で
    /// `GpuVendor::Unknown`のまま(=DXGI列挙が機能していない)になって
    /// いないか、デバイス名がプレースホルダ文字列のままになっていない
    /// かを検証する。
    #[test]
    fn real_d3d12_device_reports_a_real_adapter_name_and_known_vendor_via_dxgi() {
        let device = match DirectXDevice::new(0) {
            Ok(dev) => dev,
            Err(e) => {
                eprintln!("skipping real D3D12 test: {e}");
                return;
            }
        };
        let dev: std::sync::Arc<dyn opencuda_core::GpuDevice> = device;
        let info = dev.info();
        println!("DXGI adapter: name={:?} vendor={:?} total_memory={}", info.name, info.vendor, info.total_memory);
        assert_ne!(info.name, "DirectX 12 Device (default adapter, feature level 11_0+)", "DXGI enumeration did not run; fell back to the generic placeholder name");
        assert!(!matches!(info.vendor, opencuda_core::GpuVendor::Unknown), "DXGI enumeration did not resolve a known vendor ID");
    }

    /// 実機でのDXILカーネルディスパッチ(Phase 2)。事前コンパイル済み
    /// `vector_add.dxil`が無い場合(`tools/compile-dx12-shaders.sh`未実行)
    /// は`eprintln!`でスキップする(Vulkanの`matmul_vulkan_real`テストと
    /// 同じパターン)。
    #[test]
    fn real_d3d12_dispatches_vector_add_and_matches_cpu_reference() {
        let dxil_path = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/vector_add.dxil");
        let dxil = match std::fs::read(dxil_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("skipping real D3D12 dispatch test: cannot read {dxil_path}: {e} (run tools/compile-dx12-shaders.sh first)");
                return;
            }
        };

        let device = match DirectXDevice::new(0) {
            Ok(dev) => dev,
            Err(e) => {
                eprintln!("skipping real D3D12 dispatch test: {e}");
                return;
            }
        };
        let dev: std::sync::Arc<dyn opencuda_core::GpuDevice> = device;

        let n = 256usize;
        let av: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let bv: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();
        let expected: Vec<f32> = av.iter().zip(bv.iter()).map(|(a, b)| a + b).collect();

        let a = alloc_buffer(&dev, n * 4).unwrap();
        let b = alloc_buffer(&dev, n * 4).unwrap();
        let c = alloc_buffer(&dev, n * 4).unwrap();
        a.copy_from_host(f32_slice_as_bytes(&av)).unwrap();
        b.copy_from_host(f32_slice_as_bytes(&bv)).unwrap();

        let kernel = opencuda_core::CompiledKernel::dxil("vector_add", "main", dxil);
        let cfg = opencuda_core::LaunchConfig::linear(n as u32, 64);
        dev.launch_kernel(
            &kernel,
            &cfg,
            &[
                opencuda_core::KernelArg::Ptr(a.as_ptr()),
                opencuda_core::KernelArg::Ptr(b.as_ptr()),
                opencuda_core::KernelArg::Ptr(c.as_ptr()),
                opencuda_core::KernelArg::Usize(n),
            ],
        )
        .unwrap();

        let mut out = vec![0u8; n * 4];
        c.copy_to_host(&mut out).unwrap();
        let result: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect();

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-3, "GPU result {r} does not match CPU reference {e}");
        }
    }

    /// 実機でのmatmul(行優先、C = A × B)ディスパッチ。`vector_add.dxil`
    /// と同じくコンパイル済み`matmul.dxil`が無い場合はスキップする。
    #[test]
    fn real_d3d12_dispatches_matmul_and_matches_cpu_reference() {
        let dxil_path = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/matmul.dxil");
        let dxil = match std::fs::read(dxil_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("skipping real D3D12 matmul test: cannot read {dxil_path}: {e} (run tools/compile-dx12-shaders.sh first)");
                return;
            }
        };

        let device = match DirectXDevice::new(0) {
            Ok(dev) => dev,
            Err(e) => {
                eprintln!("skipping real D3D12 matmul test: {e}");
                return;
            }
        };
        let dev: std::sync::Arc<dyn opencuda_core::GpuDevice> = device;

        // A: 4x3, B: 3x5, C: 4x5 (行優先)。
        let (m, k, n) = (4usize, 3usize, 5usize);
        let av: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.5).collect();
        let bv: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.25).collect();
        let mut expected = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for i in 0..k {
                    sum += av[row * k + i] * bv[i * n + col];
                }
                expected[row * n + col] = sum;
            }
        }

        let a = alloc_buffer(&dev, m * k * 4).unwrap();
        let b = alloc_buffer(&dev, k * n * 4).unwrap();
        let c = alloc_buffer(&dev, m * n * 4).unwrap();
        a.copy_from_host(f32_slice_as_bytes(&av)).unwrap();
        b.copy_from_host(f32_slice_as_bytes(&bv)).unwrap();

        let kernel = opencuda_core::CompiledKernel::dxil("matmul", "main", dxil);
        let cfg = opencuda_core::LaunchConfig::grid2d(m as u32, n as u32, 8, 8);
        dev.launch_kernel(
            &kernel,
            &cfg,
            &[
                opencuda_core::KernelArg::Ptr(a.as_ptr()),
                opencuda_core::KernelArg::Ptr(b.as_ptr()),
                opencuda_core::KernelArg::Ptr(c.as_ptr()),
                opencuda_core::KernelArg::Usize(m),
                opencuda_core::KernelArg::Usize(k),
                opencuda_core::KernelArg::Usize(n),
            ],
        )
        .unwrap();

        let mut out = vec![0u8; m * n * 4];
        c.copy_to_host(&mut out).unwrap();
        let result: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect();

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-3, "GPU matmul result {r} does not match CPU reference {e}");
        }
    }

    fn f32_slice_as_bytes(values: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
    }

    /// 実機でのChaCha20ディスパッチ。RustCrypto製`chacha20`クレート
    /// (CPU参照実装)と同一の鍵・ノンス・カウンタ・平文で暗号化した
    /// 結果がバイト単位で完全一致することを検証する(自己記憶頼りの
    /// RFCテストベクタ転記ミスのリスクを避けるため、信頼できる既存
    /// 実装との突き合わせという方法を採用)。
    #[test]
    fn real_d3d12_dispatches_chacha20_and_matches_rustcrypto_reference() {
        use chacha20::cipher::{KeyIvInit, StreamCipher};
        use chacha20::ChaCha20;

        let dxil_path = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/chacha20.dxil");
        let dxil = match std::fs::read(dxil_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("skipping real D3D12 chacha20 test: cannot read {dxil_path}: {e} (run tools/compile-dx12-shaders.sh first)");
                return;
            }
        };

        let device = match DirectXDevice::new(0) {
            Ok(dev) => dev,
            Err(e) => {
                eprintln!("skipping real D3D12 chacha20 test: {e}");
                return;
            }
        };
        let dev: std::sync::Arc<dyn opencuda_core::GpuDevice> = device;

        let key_bytes: [u8; 32] = std::array::from_fn(|i| i as u8);
        let nonce_bytes: [u8; 12] = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00];
        let plaintext = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it. Extra padding to span multiple 64-byte ChaCha20 blocks for a more thorough test!!";

        // CPU参照実装(RustCrypto)
        let mut cpu_buf = plaintext.to_vec();
        // 64バイト境界に揃える(GPU側はブロック単位でのみ処理するため、
        // 端数が出ないようテストデータ自体を4の倍数長にしてある)。
        let mut cipher = ChaCha20::new(&key_bytes.into(), &nonce_bytes.into());
        cipher.apply_keystream(&mut cpu_buf);

        // GPU実装
        let mut key_words = [0u32; 8];
        for (i, chunk) in key_bytes.chunks_exact(4).enumerate() {
            key_words[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        let mut nonce_words = [0u32; 3];
        for (i, chunk) in nonce_bytes.chunks_exact(4).enumerate() {
            nonce_words[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }

        // データ長を4バイト境界へパディング(GPUカーネルはu32ワード単位)。
        let mut gpu_buf = plaintext.to_vec();
        while !gpu_buf.len().is_multiple_of(4) {
            gpu_buf.push(0);
        }

        let data = alloc_buffer(&dev, gpu_buf.len()).unwrap();
        data.copy_from_host(&gpu_buf).unwrap();

        let kernel = opencuda_core::CompiledKernel::dxil("chacha20", "main", dxil);
        let cfg = opencuda_core::LaunchConfig::linear(1, 1); // dispatch_chacha20は独自にgroup countを計算するため未使用
        let mut launch_args = vec![opencuda_core::KernelArg::Ptr(data.as_ptr())];
        for k in key_words {
            launch_args.push(opencuda_core::KernelArg::U32(k));
        }
        for n in nonce_words {
            launch_args.push(opencuda_core::KernelArg::U32(n));
        }
        launch_args.push(opencuda_core::KernelArg::U32(0)); // counter_base = 0 (RustCrypto ChaCha20::newの既定開始位置と一致させる)

        dev.launch_kernel(&kernel, &cfg, &launch_args).unwrap();

        let mut gpu_result = vec![0u8; gpu_buf.len()];
        data.copy_to_host(&mut gpu_result).unwrap();

        assert_eq!(&gpu_result[..plaintext.len()], &cpu_buf[..], "GPU ChaCha20 output does not match RustCrypto CPU reference");
    }

    /// 実機でのPoly1305バッチディスパッチ(2026-07-30追記、前回HANDOFFの
    /// 「実装難度が高く見送り」を実際に解消した実装)。RustCrypto製
    /// `poly1305`クレート(CPU参照実装)と同一の鍵・メッセージで計算した
    /// タグがバイト単位で完全一致することを検証する(ChaCha20テストと
    /// 同じ「信頼できる既存実装との突き合わせ」方針)。複数の独立した
    /// メッセージを1回のディスパッチでバッチ処理できることも合わせて
    /// 検証する(RS-LinkFusionの多数の独立パケットという実利用形態を
    /// 想定した設計、`poly1305.hlsl`冒頭のdocコメント参照)。
    #[test]
    fn real_d3d12_dispatches_poly1305_batch_and_matches_rustcrypto_reference() {
        use poly1305::universal_hash::KeyInit;
        use poly1305::{Key, Poly1305};

        let dxil_path = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/poly1305.dxil");
        let dxil = match std::fs::read(dxil_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("skipping real D3D12 poly1305 test: cannot read {dxil_path}: {e} (run tools/compile-dx12-shaders.sh first)");
                return;
            }
        };

        let device = match DirectXDevice::new(0) {
            Ok(dev) => dev,
            Err(e) => {
                eprintln!("skipping real D3D12 poly1305 test: {e}");
                return;
            }
        };
        let dev: std::sync::Arc<dyn opencuda_core::GpuDevice> = device;

        // 3個の独立したメッセージをバッチ処理する(鍵・長さともに異なる、
        // 16バイト境界の整数倍長。GPU実装の「メッセージ長は16バイトの
        // 整数倍のみ対応」という正直な制約に合わせたテストデータ)。
        const NUM_MESSAGES: usize = 3;
        const MAX_BLOCKS: usize = 4; // 最大64バイト/メッセージ
        let keys: Vec<[u8; 32]> = (0..NUM_MESSAGES)
            .map(|m| std::array::from_fn(|i| ((m * 7 + i * 3 + 11) % 256) as u8))
            .collect();
        let messages: Vec<Vec<u8>> = vec![
            b"Cryptographic Forum Research Group".to_vec(),
            b"RS-LinkFusion GPU Poly1305 batch test payload!!".to_vec(),
            b"short block xx16".to_vec(),
        ];
        // 16バイト境界に切り詰める(GPU実装は端数ブロック未対応のため)。
        let messages: Vec<Vec<u8>> = messages.into_iter().map(|m| { let n = (m.len() / 16) * 16; m[..n].to_vec() }).collect();

        // CPU参照実装(RustCrypto)
        let expected_tags: Vec<[u8; 16]> = keys
            .iter()
            .zip(messages.iter())
            .map(|(key, msg)| {
                let poly = Poly1305::new(Key::from_slice(key));
                let tag = poly.compute_unpadded(msg);
                let mut out = [0u8; 16];
                out.copy_from_slice(tag.as_slice());
                out
            })
            .collect();

        // GPU側バッファ構築
        let mut data_words = vec![0u32; NUM_MESSAGES * MAX_BLOCKS * 4];
        let mut key_words = vec![0u32; NUM_MESSAGES * 8];
        let mut block_counts = vec![0u32; NUM_MESSAGES];
        for (m, (key, msg)) in keys.iter().zip(messages.iter()).enumerate() {
            for (i, chunk) in key.chunks_exact(4).enumerate() {
                key_words[m * 8 + i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            let blocks = msg.len() / 16;
            assert!(blocks <= MAX_BLOCKS, "test message longer than MAX_BLOCKS");
            block_counts[m] = blocks as u32;
            for (j, chunk) in msg.chunks_exact(4).enumerate() {
                data_words[m * MAX_BLOCKS * 4 + j] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }

        let data_buf = alloc_buffer(&dev, data_words.len() * 4).unwrap();
        let keys_buf = alloc_buffer(&dev, key_words.len() * 4).unwrap();
        let counts_buf = alloc_buffer(&dev, block_counts.len() * 4).unwrap();
        let tags_buf = alloc_buffer(&dev, NUM_MESSAGES * 16).unwrap();
        data_buf.copy_from_host(u32_slice_as_bytes(&data_words)).unwrap();
        keys_buf.copy_from_host(u32_slice_as_bytes(&key_words)).unwrap();
        counts_buf.copy_from_host(u32_slice_as_bytes(&block_counts)).unwrap();

        let kernel = opencuda_core::CompiledKernel::dxil("poly1305", "main", dxil);
        let cfg = opencuda_core::LaunchConfig::linear(1, 1); // dispatch_poly1305は独自にgroup countを計算するため未使用
        dev.launch_kernel(
            &kernel,
            &cfg,
            &[
                opencuda_core::KernelArg::Ptr(data_buf.as_ptr()),
                opencuda_core::KernelArg::Ptr(keys_buf.as_ptr()),
                opencuda_core::KernelArg::Ptr(counts_buf.as_ptr()),
                opencuda_core::KernelArg::Ptr(tags_buf.as_ptr()),
                opencuda_core::KernelArg::U32(MAX_BLOCKS as u32),
            ],
        )
        .unwrap();

        let mut tags_out = vec![0u8; NUM_MESSAGES * 16];
        tags_buf.copy_to_host(&mut tags_out).unwrap();

        for (m, expected) in expected_tags.iter().enumerate() {
            let got = &tags_out[m * 16..m * 16 + 16];
            assert_eq!(got, expected, "GPU Poly1305 tag for message {m} does not match RustCrypto CPU reference");
        }
    }

    fn u32_slice_as_bytes(values: &[u32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencuda_core::alloc_buffer;

    #[test]
    fn mock_device_rejects_non_dxil_kernel_source() {
        let dev: Arc<dyn GpuDevice> = DirectXMockDevice::new(0);
        let kernel = CompiledKernel::native("noop", |_ctx, _args| {});
        let cfg = LaunchConfig::linear(1, 1);
        let err = dev.launch_kernel(&kernel, &cfg, &[]).unwrap_err();
        assert!(err.to_string().contains("not supported"));
    }

    #[test]
    fn mock_device_rejects_dxil_without_dxbc_magic() {
        let dev: Arc<dyn GpuDevice> = DirectXMockDevice::new(0);
        let kernel = CompiledKernel::dxil("vector_add", "main", vec![0u8; 8]);
        let cfg = LaunchConfig::linear(1, 1);
        let err = dev.launch_kernel(&kernel, &cfg, &[]).unwrap_err();
        assert!(err.to_string().contains("DXBC"));
    }

    #[test]
    fn mock_device_simulates_vector_add_via_dxil_path() {
        let dev: Arc<dyn GpuDevice> = DirectXMockDevice::new(0);
        assert!(dev.supports_dxil());

        let n = 4usize;
        let a = alloc_buffer(&dev, n * 4).unwrap();
        let b = alloc_buffer(&dev, n * 4).unwrap();
        let c = alloc_buffer(&dev, n * 4).unwrap();

        let av = [1.0f32, 2.0, 3.0, 4.0];
        let bv = [10.0f32, 20.0, 30.0, 40.0];
        a.copy_from_host(bytemuck_cast_f32(&av)).unwrap();
        b.copy_from_host(bytemuck_cast_f32(&bv)).unwrap();

        let mut dxil = DXBC_MAGIC.to_vec();
        dxil.extend_from_slice(&[0u8; 4]);
        let kernel = CompiledKernel::dxil("vector_add", "main", dxil);
        let cfg = LaunchConfig::linear(n as u32, 1);
        dev.launch_kernel(
            &kernel,
            &cfg,
            &[KernelArg::Ptr(a.as_ptr()), KernelArg::Ptr(b.as_ptr()), KernelArg::Ptr(c.as_ptr()), KernelArg::Usize(n)],
        )
        .unwrap();

        let mut out = vec![0u8; n * 4];
        c.copy_to_host(&mut out).unwrap();
        let result: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    fn bytemuck_cast_f32(values: &[f32]) -> &[u8] {
        // no bytemuck dependency; manual reinterpret via slice of bytes
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
    }
}
