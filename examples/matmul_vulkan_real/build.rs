fn main() {
    opencuda_shader_build::compile_glsl_shaders(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders"));
}
