//! vector_add: OpenCUDA Phase 1 の最小サンプル。
//!
//! C = A + B を CPU バックエンドで実行する。CUDA の
//! `__global__ void add(float* a, float* b, float* c, int n)` に相当する
//! カーネルを Rust の Native カーネルとして書く。
//!
//! 実行: `cargo run --release --example vector_add` あるいは
//!       このバイナリを直接 `cargo run -p vector_add`

use std::sync::Arc;

use anyhow::Result;
use opencuda_core::{
    alloc_buffer, CompiledKernel, GpuDevice, KernelArg, LaunchConfig, ResolvedArg, ThreadCtx,
};
use opencuda_cpu::CpuDevice;

const N: usize = 1_000_000;

fn main() -> Result<()> {
    // 1. デバイスを用意（CPUバックエンド）。
    let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
    println!("device: {}", device.info().name);

    // 2. ホスト側の入力を用意。
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();

    let bytes = N * std::mem::size_of::<f32>();

    // 3. デバイスメモリを確保（RAII バッファ）。
    let da = alloc_buffer(&device, bytes)?;
    let db = alloc_buffer(&device, bytes)?;
    let dc = alloc_buffer(&device, bytes)?;

    // 4. ホスト → デバイス転送。
    da.copy_from_host(bytemuck_cast(&a))?;
    db.copy_from_host(bytemuck_cast(&b))?;

    // 5. カーネル定義。1スレッド = 1要素。CUDA の add カーネルに対応。
    let kernel = CompiledKernel::native("vector_add", |ctx: ThreadCtx, args: &[ResolvedArg]| {
        let i = ctx.global_id_x() as usize;
        let (a_ptr, a_len) = args[0].as_ptr().unwrap();
        let (b_ptr, _) = args[1].as_ptr().unwrap();
        let (c_ptr, _) = args[2].as_ptr().unwrap();
        let n = args[3].as_usize().unwrap();

        if i >= n {
            return;
        }
        // 範囲ガード（a_len はバイト数）。
        debug_assert!((i + 1) * 4 <= a_len);

        // SAFETY: i < n かつ各バッファは n*4 バイト確保済み。各スレッドは
        // 自分の i のみ書くため競合しない。
        unsafe {
            let a = (a_ptr as *const f32).add(i).read();
            let b = (b_ptr as *const f32).add(i).read();
            (c_ptr as *mut f32).add(i).write(a + b);
        }
    });

    // 6. 起動設定（1次元、ブロックサイズ256）。
    let cfg = LaunchConfig::linear(N as u32, 256);

    // 7. 引数を渡して起動。
    device.launch_kernel(
        &kernel,
        &cfg,
        &[
            KernelArg::Ptr(da.as_ptr()),
            KernelArg::Ptr(db.as_ptr()),
            KernelArg::Ptr(dc.as_ptr()),
            KernelArg::Usize(N),
        ],
    )?;
    device.synchronize()?;

    // 8. デバイス → ホスト転送。
    let mut c = vec![0.0f32; N];
    dc.copy_to_host(bytemuck_cast_mut(&mut c))?;

    // 9. 検証。A[i] + B[i] = i + (N - i) = N（全要素同じ）。
    let expected = N as f32;
    let mut ok = true;
    for (idx, &v) in c.iter().enumerate() {
        if (v - expected).abs() > 1e-3 {
            eprintln!("mismatch at {idx}: got {v}, expected {expected}");
            ok = false;
            break;
        }
    }

    if ok {
        println!("OK: all {N} elements equal {expected}");
        println!("c[0]={}, c[{}]={}", c[0], N - 1, c[N - 1]);
        Ok(())
    } else {
        anyhow::bail!("verification failed")
    }
}

// 依存を増やさないための最小キャスト（&[f32] -> &[u8]）。
// f32 は無条件にバイト列として読めるので安全。
fn bytemuck_cast(v: &[f32]) -> &[u8] {
    // SAFETY: f32 スライスを read-only な u8 スライスとして見るだけ。
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytemuck_cast_mut(v: &mut [f32]) -> &mut [u8] {
    // SAFETY: 同上、可変版。アライメントは f32 の方が厳しいので u8 化は安全。
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}
