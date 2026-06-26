//! matmul: CPUバックエンドで小さな行列積 C = A·B を検証するサンプル。

use std::sync::Arc;

use anyhow::Result;
use opencuda_core::{
    alloc_buffer, CompiledKernel, GpuDevice, KernelArg, LaunchConfig, ResolvedArg, ThreadCtx,
};
use opencuda_cpu::CpuDevice;

const M: usize = 64;
const K: usize = 64;
const N: usize = 64;

fn main() -> Result<()> {
    let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
    println!("device: {}", device.info().name);

    let a: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32).collect();
    let b: Vec<f32> = (0..K * N).map(|i| (i % 5) as f32).collect();
    let mut c_ref = vec![0.0f32; M * N];

    for row in 0..M {
        for col in 0..N {
            let mut acc = 0.0f32;
            for kk in 0..K {
                acc += a[row * K + kk] * b[kk * N + col];
            }
            c_ref[row * N + col] = acc;
        }
    }

    let da = alloc_buffer(&device, M * K * 4)?;
    let db = alloc_buffer(&device, K * N * 4)?;
    let dc = alloc_buffer(&device, M * N * 4)?;
    da.copy_from_host(cast_f32_to_u8(&a))?;
    db.copy_from_host(cast_f32_to_u8(&b))?;

    let kernel = CompiledKernel::native("matmul_naive", |ctx: ThreadCtx, args: &[ResolvedArg]| {
        let idx = ctx.global_id_x() as usize;
        let m = args[3].as_usize().unwrap();
        let k = args[4].as_usize().unwrap();
        let n = args[5].as_usize().unwrap();
        if idx >= m * n {
            return;
        }
        let row = idx / n;
        let col = idx % n;
        let (a_ptr, _) = args[0].as_ptr().unwrap();
        let (b_ptr, _) = args[1].as_ptr().unwrap();
        let (c_ptr, _) = args[2].as_ptr().unwrap();

        let mut acc = 0.0f32;
        unsafe {
            let a = a_ptr as *const f32;
            let b = b_ptr as *const f32;
            let c = c_ptr as *mut f32;
            for kk in 0..k {
                acc += a.add(row * k + kk).read() * b.add(kk * n + col).read();
            }
            c.add(idx).write(acc);
        }
    });

    let cfg = LaunchConfig::linear((M * N) as u32, 256);
    device.launch_kernel(
        &kernel,
        &cfg,
        &[
            KernelArg::Ptr(da.as_ptr()),
            KernelArg::Ptr(db.as_ptr()),
            KernelArg::Ptr(dc.as_ptr()),
            KernelArg::Usize(M),
            KernelArg::Usize(K),
            KernelArg::Usize(N),
        ],
    )?;
    device.synchronize()?;

    let mut c = vec![0.0f32; M * N];
    dc.copy_to_host(cast_f32_to_u8_mut(&mut c))?;

    for (idx, (&got, &expected)) in c.iter().zip(c_ref.iter()).enumerate() {
        if (got - expected).abs() > 1e-3 {
            anyhow::bail!("mismatch at {idx}: got {got}, expected {expected}");
        }
    }

    println!("OK: matmul {M}x{K} * {K}x{N} verified");
    Ok(())
}

fn cast_f32_to_u8(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn cast_f32_to_u8_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}
