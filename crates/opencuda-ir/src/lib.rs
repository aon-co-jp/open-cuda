//! # opencuda-ir
//!
//! OpenCUDA v0.2 の最小 OmniIR 実装。
//!
//! 目的は「CUDA全体のIR」ではなく、まず `vector_add_f32` を IR として表現し、
//! CPU Native と Vulkan/SPIR-V 経路へ同じ意味を流せることを確認すること。
//! これにより、GPU実機が無くてもフロントエンド・IR・バックエンド契約のBUGを潰せる。

use anyhow::{anyhow, bail, Result};
use opencuda_core::{CompiledKernel, ResolvedArg, ThreadCtx};

const MAGIC: &[u8] = b"OCIR2\0";
const VECTOR_ADD_F32: &str = "vector_add_f32";

/// v0.2 の最小IRモジュール。
///
/// 今は `vector_add_f32` だけを表す。将来ここに typed SSA、shared memory、barrier、
/// warp/ subgroup 命令を足していく。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrModule {
    pub name: String,
    pub entry: String,
    pub ops: Vec<IrOp>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IrOp {
    /// c[i] = a[i] + b[i], i = global_id_x, guard: i < n
    VectorAddF32 {
        a_arg: usize,
        b_arg: usize,
        c_arg: usize,
        n_arg: usize,
    },
}

impl IrModule {
    /// OpenCUDA v0.2 の基準カーネル。
    pub fn vector_add_f32() -> Self {
        Self {
            name: VECTOR_ADD_F32.to_string(),
            entry: "main".to_string(),
            ops: vec![IrOp::VectorAddF32 {
                a_arg: 0,
                b_arg: 1,
                c_arg: 2,
                n_arg: 3,
            }],
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        // v0.2 の安定fixture。意図的に単純化し、人間が読める形式にしている。
        // 将来は postcard / bincode / custom binary IR に置き換える。
        let mut out = Vec::from(MAGIC);
        out.extend_from_slice(self.name.as_bytes());
        out.push(0);
        out.extend_from_slice(self.entry.as_bytes());
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if !bytes.starts_with(MAGIC) {
            bail!("invalid OmniIR: missing OCIR2 magic");
        }
        let rest = &bytes[MAGIC.len()..];
        let Some(pos) = rest.iter().position(|&b| b == 0) else {
            bail!("invalid OmniIR: missing name separator");
        };
        let name = std::str::from_utf8(&rest[..pos])?.to_string();
        let entry = std::str::from_utf8(&rest[pos + 1..])?.to_string();
        match name.as_str() {
            VECTOR_ADD_F32 => Ok(Self::vector_add_f32()),
            other => Err(anyhow!("unsupported OmniIR module `{other}` in v0.2")),
        }
        .map(|mut m| {
            m.entry = entry;
            m
        })
    }

    pub fn to_compiled_omniir(&self) -> CompiledKernel {
        CompiledKernel::omniir(self.name.clone(), self.entry.clone(), self.encode())
    }
}

/// OmniIRをCPU Nativeカーネルへ下げる。
///
/// v0.2では `vector_add_f32` のみ。将来は命令列インタプリタまたはJITへ拡張する。
pub fn lower_to_native(module: &IrModule) -> Result<CompiledKernel> {
    if module.ops.as_slice()
        != &[IrOp::VectorAddF32 {
            a_arg: 0,
            b_arg: 1,
            c_arg: 2,
            n_arg: 3,
        }]
    {
        bail!("opencuda-ir v0.2 only lowers canonical vector_add_f32");
    }

    Ok(CompiledKernel::native(module.name.clone(), |ctx: ThreadCtx, args: &[ResolvedArg]| {
        let i = ctx.global_id_x() as usize;
        let (a_ptr, a_len) = args[0].as_ptr().expect("arg0 pointer");
        let (b_ptr, b_len) = args[1].as_ptr().expect("arg1 pointer");
        let (c_ptr, c_len) = args[2].as_ptr().expect("arg2 pointer");
        let n = args[3].as_usize().expect("arg3 usize");

        if i >= n {
            return;
        }
        let need = (i + 1) * std::mem::size_of::<f32>();
        debug_assert!(need <= a_len && need <= b_len && need <= c_len);

        // SAFETY: caller allocated at least n*f32 bytes and each logical thread writes only c[i].
        unsafe {
            let a = (a_ptr as *const f32).add(i).read();
            let b = (b_ptr as *const f32).add(i).read();
            (c_ptr as *mut f32).add(i).write(a + b);
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_add_roundtrip() {
        let module = IrModule::vector_add_f32();
        let encoded = module.encode();
        let decoded = IrModule::decode(&encoded).unwrap();
        assert_eq!(decoded.name, "vector_add_f32");
        assert_eq!(decoded.entry, "main");
    }
}
