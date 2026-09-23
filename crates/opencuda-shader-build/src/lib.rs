//! 各クレート/exampleの`build.rs`から呼ぶ共通ヘルパー(2026-09-23新設)。
//!
//! **見つけたバグ**: `examples/*_vulkan_real`系(hgemm/dgemm/matmul/softmax/raid6等、
//! 計9例)は`shaders/*.comp`を`glslc`で`*.spv`へ手動コンパイルする前提のまま
//! `build.rs`が一つも無く、`.spv`自体は`.gitignore`(`**/*.spv`)でGit管理外だった。
//! そのためクリーンチェックアウト直後(新しい開発者・CI)は必ず
//! `include_bytes!`/実行時ロードの両方で「ファイルが見つからない」失敗になる
//! ——実際に`cargo test --workspace`実行時に`opencuda-blas`のテストで再現した。
//! 個々の`build.rs`に同じ処理を重複させず、この1クレートへ集約することで、
//! 新しいVulkan例を追加するたびに同じ落とし穴を踏まないようにする。
//!
//! **正直な開示**: `glslc`(Vulkan SDK付属)がPATHに無い環境では、分かりやすい
//! エラーメッセージ付きでビルド自体を失敗させる(既存カーネルが動くふりをして
//! 実は古い/存在しない`.spv`を握ったまま実行時に不可解な失敗をする方が悪いため)。

use std::path::Path;
use std::process::Command;

/// `shader_dir`直下の`*.comp`をすべて同じディレクトリへ`*.spv`としてコンパイルする。
/// `.comp`を編集すれば次のビルドで再コンパイルされるよう`cargo:rerun-if-changed`も出す。
pub fn compile_glsl_shaders(shader_dir: impl AsRef<Path>) {
    let dir = shader_dir.as_ref();
    println!("cargo:rerun-if-changed={}", dir.display());
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => panic!("opencuda-shader-build: failed to read shader dir {}: {e}", dir.display()),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("comp") {
            continue;
        }
        compile_one(&path, &path.with_extension("spv"));
    }
}

/// 単一ファイル版(`opencuda-blas`のように、自クレート外〈`examples/`〉の
/// シェーダーソースを参照する特殊なケース向け)。
pub fn compile_one(comp_path: impl AsRef<Path>, spv_path: impl AsRef<Path>) {
    let comp_path = comp_path.as_ref();
    let spv_path = spv_path.as_ref();
    println!("cargo:rerun-if-changed={}", comp_path.display());
    let status = Command::new("glslc").arg(comp_path).arg("-o").arg(spv_path).status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("opencuda-shader-build: glslc failed compiling {} (exit {s})", comp_path.display()),
        Err(e) => panic!(
            "opencuda-shader-build: could not run `glslc` to compile {} — is the Vulkan SDK installed and is glslc on PATH? ({e})",
            comp_path.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_one_produces_readable_spirv_for_valid_shader() {
        let tmp = std::env::temp_dir().join(format!("opencuda_shader_build_test_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let comp = tmp.join("t.comp");
        std::fs::write(
            &comp,
            "#version 450\nlayout(local_size_x = 1) in;\nvoid main() {}\n",
        )
        .unwrap();
        let spv = tmp.join("t.spv");
        compile_one(&comp, &spv);
        assert!(spv.exists(), "expected {} to be produced by glslc", spv.display());
        let bytes = std::fs::read(&spv).unwrap();
        assert!(!bytes.is_empty());
        // SPIR-Vマジックナンバー(リトルエンディアン 0x07230203)を確認。
        assert_eq!(&bytes[0..4], &[0x03, 0x02, 0x23, 0x07]);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
