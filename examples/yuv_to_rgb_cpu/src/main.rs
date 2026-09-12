//! yuv_to_rgb_cpu: 動画パイプライン向けGPUカーネルの試作第一弾。
//!
//! **重要な位置付け(正直な開示)**: これは動画コーデック(H.264等)の
//! 実装ではない。コーデックの実装(動き予測・DCT変換・量子化・
//! エントロピー符号化・レート制御)は本アプリ単体で扱える規模を
//! はるかに超えるため、対象外とする。ここで扱うのは、動画パイプラインで
//! 頻出する「ピクセルフォーマット変換」という一処理のみ:
//! I420(YUV420 planar) → RGB24(interleaved)を、OpenCUDAのCPU
//! バックエンド(rayon並列)上のカーネルとして実装し、参照実装(スカラー
//! ループ)と数値一致することを検証する。
//!
//! 「小さな試作から少しずつ大きくする」方針の第一歩として、まずCPU
//! バックエンドのみで実装・検証する(GPU実機ディスパッチは次段階)。
//!
//! 実行: `cargo run --release --example yuv_to_rgb_cpu`
//!       あるいは `cargo run -p yuv_to_rgb_cpu`

use std::sync::Arc;

use anyhow::Result;
use opencuda_core::{alloc_buffer, CompiledKernel, GpuDevice, KernelArg, LaunchConfig, ResolvedArg, ThreadCtx};
use opencuda_cpu::CpuDevice;

const WIDTH: usize = 64;
const HEIGHT: usize = 48;

fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// BT.601変換式によるスカラー参照実装(1画素分)。GPU/CPUカーネル側の
/// 出力がこれと完全一致することを検証の基準にする。
fn yuv_to_rgb_reference(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let yf = y as f32;
    let uf = u as f32 - 128.0;
    let vf = v as f32 - 128.0;
    let r = yf + 1.402 * vf;
    let g = yf - 0.344136 * uf - 0.714136 * vf;
    let b = yf + 1.772 * uf;
    (clamp_u8(r), clamp_u8(g), clamp_u8(b))
}

fn main() -> Result<()> {
    let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
    println!("device: {}", device.info().name);

    // I420(YUV420 planar)のテスト画像を生成: グラデーションパターン。
    let y_size = WIDTH * HEIGHT;
    let chroma_w = WIDTH / 2;
    let chroma_h = HEIGHT / 2;
    let chroma_size = chroma_w * chroma_h;

    let mut yuv = vec![0u8; y_size + 2 * chroma_size];
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            yuv[row * WIDTH + col] = ((row * 3 + col * 5) % 256) as u8;
        }
    }
    for row in 0..chroma_h {
        for col in 0..chroma_w {
            yuv[y_size + row * chroma_w + col] = ((row * 7 + col * 2 + 40) % 256) as u8;
            yuv[y_size + chroma_size + row * chroma_w + col] = ((row * 2 + col * 11 + 90) % 256) as u8;
        }
    }

    let device_in = alloc_buffer(&device, yuv.len())?;
    device_in.copy_from_host(&yuv)?;

    let rgb_len = y_size * 3;
    let device_out = alloc_buffer(&device, rgb_len)?;

    let kernel = CompiledKernel::native("yuv420p_to_rgb24", |ctx: ThreadCtx, args: &[ResolvedArg]| {
        let idx = ctx.global_id_x() as usize; // 0..width*height の画素インデックス
        let width = args[2].as_usize().unwrap();
        let height = args[3].as_usize().unwrap();
        let n = width * height;
        if idx >= n {
            return;
        }
        let row = idx / width;
        let col = idx % width;
        let chroma_w = width / 2;
        let chroma_size = chroma_w * (height / 2);

        let (in_ptr, _in_len) = args[0].as_ptr().unwrap();
        let (out_ptr, _out_len) = args[1].as_ptr().unwrap();

        // SAFETY: idx < n であり、in_ptr/out_ptrは各々yuv.len()/rgb_len
        // バイト確保済み。各スレッドは自分のidxに対応する箇所のみ読み書き
        // するため競合しない。
        unsafe {
            let y = (in_ptr as *const u8).add(row * width + col).read();
            let u = (in_ptr as *const u8).add(n + (row / 2) * chroma_w + col / 2).read();
            let v = (in_ptr as *const u8).add(n + chroma_size + (row / 2) * chroma_w + col / 2).read();

            let yf = y as f32;
            let uf = u as f32 - 128.0;
            let vf = v as f32 - 128.0;
            let r = (yf + 1.402 * vf).round().clamp(0.0, 255.0) as u8;
            let g = (yf - 0.344136 * uf - 0.714136 * vf).round().clamp(0.0, 255.0) as u8;
            let b = (yf + 1.772 * uf).round().clamp(0.0, 255.0) as u8;

            let out = (out_ptr as *mut u8).add(idx * 3);
            out.write(r);
            out.add(1).write(g);
            out.add(2).write(b);
        }
    });

    let cfg = LaunchConfig::linear((y_size) as u32, 256);
    device.launch_kernel(
        &kernel,
        &cfg,
        &[
            KernelArg::Ptr(device_in.as_ptr()),
            KernelArg::Ptr(device_out.as_ptr()),
            KernelArg::Usize(WIDTH),
            KernelArg::Usize(HEIGHT),
        ],
    )?;
    device.synchronize()?;

    let mut rgb = vec![0u8; rgb_len];
    device_out.copy_to_host(&mut rgb)?;

    // 検証: 全画素についてスカラー参照実装と完全一致するか確認する。
    let mut mismatches = 0;
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            let idx = row * WIDTH + col;
            let y = yuv[idx];
            let u = yuv[y_size + (row / 2) * chroma_w + col / 2];
            let v = yuv[y_size + chroma_size + (row / 2) * chroma_w + col / 2];
            let expected = yuv_to_rgb_reference(y, u, v);
            let got = (rgb[idx * 3], rgb[idx * 3 + 1], rgb[idx * 3 + 2]);
            if got != expected {
                if mismatches < 5 {
                    eprintln!("mismatch at ({row},{col}): got {got:?}, expected {expected:?}");
                }
                mismatches += 1;
            }
        }
    }

    if mismatches == 0 {
        println!("OK: all {} pixels match the scalar BT.601 reference implementation", WIDTH * HEIGHT);
        Ok(())
    } else {
        anyhow::bail!("{mismatches} pixel(s) mismatched");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_pixel_values_match_expected_rgb() {
        // Y=255,U=128,V=128 (中間色クロマ、最大輝度) は白に近くなるはず。
        let (r, g, b) = yuv_to_rgb_reference(255, 128, 128);
        assert_eq!((r, g, b), (255, 255, 255));

        // Y=0,U=128,V=128 は黒。
        let (r, g, b) = yuv_to_rgb_reference(0, 128, 128);
        assert_eq!((r, g, b), (0, 0, 0));
    }

    #[test]
    fn cpu_kernel_matches_scalar_reference_end_to_end() {
        // main()の検証ロジックと同じ内容を関数として再実行できるよう、
        // ここでは代表点のみ簡易再検証する(mainの詳細な全画素検証は
        // 実行バイナリ側で行う)。
        let device: Arc<dyn GpuDevice> = CpuDevice::new(0);
        let width = 4usize;
        let height = 4usize;
        let y_size = width * height;
        let chroma_w = width / 2;
        let chroma_size = chroma_w * (height / 2);
        let yuv = vec![128u8; y_size + 2 * chroma_size];

        let device_in = alloc_buffer(&device, yuv.len()).unwrap();
        device_in.copy_from_host(&yuv).unwrap();
        let device_out = alloc_buffer(&device, y_size * 3).unwrap();

        let kernel = CompiledKernel::native("yuv420p_to_rgb24_test", |ctx: ThreadCtx, args: &[ResolvedArg]| {
            let idx = ctx.global_id_x() as usize;
            let width = args[2].as_usize().unwrap();
            let height = args[3].as_usize().unwrap();
            let n = width * height;
            if idx >= n {
                return;
            }
            let row = idx / width;
            let col = idx % width;
            let chroma_w = width / 2;
            let chroma_size = chroma_w * (height / 2);
            let (in_ptr, _) = args[0].as_ptr().unwrap();
            let (out_ptr, _) = args[1].as_ptr().unwrap();
            unsafe {
                let y = (in_ptr as *const u8).add(row * width + col).read();
                let u = (in_ptr as *const u8).add(n + (row / 2) * chroma_w + col / 2).read();
                let v = (in_ptr as *const u8).add(n + chroma_size + (row / 2) * chroma_w + col / 2).read();
                let yf = y as f32;
                let uf = u as f32 - 128.0;
                let vf = v as f32 - 128.0;
                let r = (yf + 1.402 * vf).round().clamp(0.0, 255.0) as u8;
                let g = (yf - 0.344136 * uf - 0.714136 * vf).round().clamp(0.0, 255.0) as u8;
                let b = (yf + 1.772 * uf).round().clamp(0.0, 255.0) as u8;
                let out = (out_ptr as *mut u8).add(idx * 3);
                out.write(r);
                out.add(1).write(g);
                out.add(2).write(b);
            }
        });

        let cfg = LaunchConfig::linear(y_size as u32, 256);
        device
            .launch_kernel(
                &kernel,
                &cfg,
                &[
                    KernelArg::Ptr(device_in.as_ptr()),
                    KernelArg::Ptr(device_out.as_ptr()),
                    KernelArg::Usize(width),
                    KernelArg::Usize(height),
                ],
            )
            .unwrap();
        device.synchronize().unwrap();

        let mut rgb = vec![0u8; y_size * 3];
        device_out.copy_to_host(&mut rgb).unwrap();
        // 全画素Y=U=V=128(中間グレー)なので、全RGBが同一のグレー値になるはず。
        let expected = yuv_to_rgb_reference(128, 128, 128);
        for px in rgb.chunks(3) {
            assert_eq!((px[0], px[1], px[2]), expected);
        }
    }
}
