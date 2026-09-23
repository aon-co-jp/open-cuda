fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    for (comp, spv) in [
        ("../../examples/hgemm_vulkan_real/shaders/hgemm.comp", "../../examples/hgemm_vulkan_real/shaders/hgemm.spv"),
        ("../../examples/dgemm_vulkan_real/shaders/dgemm.comp", "../../examples/dgemm_vulkan_real/shaders/dgemm.spv"),
    ] {
        opencuda_shader_build::compile_one(format!("{manifest_dir}/{comp}"), format!("{manifest_dir}/{spv}"));
    }
}
