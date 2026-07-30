//! Optional real Vulkan Compute backend implemented with `ash`.
//!
//! v0.3.5 scope is intentionally small: one compute queue, host-visible storage buffers,
//! SPIR-V shader modules supplied by the caller, and the `vector_add` argument contract.
//! This keeps the first real GPU path understandable and easy to debug.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context};
use ash::{vk, Entry};
use opencuda_core::{
    CompiledKernel, DeviceInfo, DevicePtr, GpuDevice, GpuError, GpuVendor, KernelArg, KernelSource,
    LaunchConfig, Result,
};

struct VulkanAllocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    len: usize,
    mapped_size: vk::DeviceSize,
    coherent: bool,
}

unsafe impl Send for VulkanAllocation {}

/// A minimal real Vulkan Compute device.
///
/// This is not yet a high-performance backend. It is a correctness backend for v0.3.5:
/// allocate host-visible buffers, create a compute pipeline from SPIR-V, dispatch it,
/// and read results back for comparison with the CPU reference path.
pub struct VulkanDevice {
    _entry: Entry,
    instance: ash::Instance,
    _physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    command_pool: vk::CommandPool,
    info: DeviceInfo,
    device_type: vk::PhysicalDeviceType,
    api_version: u32,
    driver_version: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    allocations: Mutex<HashMap<u64, VulkanAllocation>>,
    next_handle: AtomicU64,
}

impl VulkanDevice {
    /// Create the first available Vulkan compute device.
    pub fn new(id: usize) -> Result<Arc<Self>> {
        let entry = unsafe { Entry::load().context("failed to load Vulkan loader. Install GPU driver/Vulkan Runtime or Vulkan SDK")? };

        let app_name = CString::new("OpenCUDA").unwrap();
        let engine_name = CString::new("OpenCUDA").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 3, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 3, 0))
            .api_version(vk::API_VERSION_1_1);

        let instance_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_info, None) }.context(
            "vkCreateInstance failed. Update the GPU driver, or verify the Vulkan Runtime/SDK install",
        )?;

        let physical_devices = match unsafe { instance.enumerate_physical_devices() }
            .context("vkEnumeratePhysicalDevices failed")
        {
            Ok(v) => v,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return Err(e);
            }
        };
        if physical_devices.is_empty() {
            unsafe { instance.destroy_instance(None) };
            bail!(
                "no Vulkan physical device found. Check that a GPU driver exposing Vulkan \
                 (NVIDIA/AMD/Intel) is installed, and that no other process/VM is hiding the GPU"
            );
        }

        let mut selected = None;
        let mut seen_devices: Vec<String> = Vec::with_capacity(physical_devices.len());
        for &pd in &physical_devices {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            seen_devices.push(format!("{name} ({})", device_type_str(props.device_type)));

            let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            if let Some((family_index, _)) = families
                .iter()
                .enumerate()
                .find(|(_, f)| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            {
                selected = Some((pd, family_index as u32));
                break;
            }
        }

        let (physical_device, queue_family_index) = match selected {
            Some(v) => v,
            None => {
                unsafe { instance.destroy_instance(None) };
                bail!(
                    "no Vulkan compute queue family found on any enumerated device. \
                     Enumerated devices: [{}]. A compute-capable queue family (VK_QUEUE_COMPUTE_BIT) \
                     is required; this usually means the driver only exposes a graphics/present-only \
                     queue, which is unexpected for a modern GPU driver",
                    seen_devices.join(", ")
                );
            }
        };

        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)
            .build()];
        let device_info = vk::DeviceCreateInfo::builder().queue_create_infos(&queue_info);
        let device = match unsafe { instance.create_device(physical_device, &device_info, None) }
            .context("vkCreateDevice failed")
        {
            Ok(d) => d,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return Err(e);
            }
        };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = match unsafe { device.create_command_pool(&pool_info, None) }
            .context("vkCreateCommandPool failed")
        {
            Ok(p) => p,
            Err(e) => {
                unsafe {
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                }
                return Err(e);
            }
        };

        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let vendor = vendor_from_id(props.vendor_id);
        let total_memory = estimate_device_local_memory(&memory_properties);

        Ok(Arc::new(Self {
            _entry: entry,
            instance,
            _physical_device: physical_device,
            device,
            queue,
            queue_family_index,
            command_pool,
            info: DeviceInfo {
                id,
                vendor,
                name: format!("OpenCUDA Vulkan Device ({name})"),
                total_memory,
                compute_units: 1,
            },
            device_type: props.device_type,
            api_version: props.api_version,
            driver_version: props.driver_version,
            memory_properties,
            allocations: Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
        }))
    }

    /// v0.3.6: `vulkan_info` 用の追加診断情報。
    ///
    /// `GpuDevice` トレイトの `DeviceInfo` はバックエンド共通の最小情報しか持たないため、
    /// Vulkan固有の詳細（キューファミリ、デバイス種別、APIバージョン、ドライババージョン）は
    /// `VulkanDevice` 側の専用メソッドとして公開する。
    pub fn diagnostics(&self) -> VulkanDiagnostics {
        VulkanDiagnostics {
            queue_family_index: self.queue_family_index,
            device_type: self.device_type,
            api_version: self.api_version,
            driver_version: self.driver_version,
        }
    }

    fn get_allocation(&self, ptr: DevicePtr) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8, usize, vk::DeviceSize, bool)> {
        if ptr.device_id as usize != self.info.id {
            return Err(GpuError::InvalidPtr(ptr).into());
        }
        let map = self.allocations.lock().unwrap();
        let a = map.get(&ptr.addr).ok_or(GpuError::InvalidPtr(ptr))?;
        Ok((a.buffer, a.memory, a.mapped, a.len, a.mapped_size, a.coherent))
    }

    fn find_memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Result<u32> {
        for i in 0..self.memory_properties.memory_type_count {
            let supported = (bits & (1 << i)) != 0;
            let has_flags = self.memory_properties.memory_types[i as usize]
                .property_flags
                .contains(flags);
            if supported && has_flags {
                return Ok(i);
            }
        }
        bail!("no compatible Vulkan memory type for flags 0x{:x}", flags.as_raw())
    }

    /// Prefer HOST_VISIBLE | HOST_COHERENT, but fall back to HOST_VISIBLE.
    /// Some Vulkan stacks do not expose a coherent memory type for every buffer requirement.
    /// In that case v0.3.5 explicitly flushes host writes and invalidates host reads.
    fn find_host_visible_memory_type(&self, bits: u32) -> Result<(u32, bool)> {
        let coherent_flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        if let Ok(index) = self.find_memory_type(bits, coherent_flags) {
            return Ok((index, true));
        }
        let index = self.find_memory_type(bits, vk::MemoryPropertyFlags::HOST_VISIBLE)?;
        Ok((index, false))
    }

    fn flush_if_needed(&self, memory: vk::DeviceMemory, _mapped_size: vk::DeviceSize, coherent: bool) -> Result<()> {
        if coherent {
            return Ok(());
        }
        // Use VK_WHOLE_SIZE so the range stays valid even when the nonCoherentAtomSize
        // alignment is larger than the requested copy length. The allocation is mapped
        // from offset 0 to its full memory-requirements size.
        let range = vk::MappedMemoryRange::builder()
            .memory(memory)
            .offset(0)
            .size(vk::WHOLE_SIZE)
            .build();
        unsafe { self.device.flush_mapped_memory_ranges(&[range]) }
            .context("vkFlushMappedMemoryRanges failed")?;
        Ok(())
    }

    fn invalidate_if_needed(&self, memory: vk::DeviceMemory, _mapped_size: vk::DeviceSize, coherent: bool) -> Result<()> {
        if coherent {
            return Ok(());
        }
        // Use VK_WHOLE_SIZE for the same reason as flush_if_needed.
        let range = vk::MappedMemoryRange::builder()
            .memory(memory)
            .offset(0)
            .size(vk::WHOLE_SIZE)
            .build();
        unsafe { self.device.invalidate_mapped_memory_ranges(&[range]) }
            .context("vkInvalidateMappedMemoryRanges failed")?;
        Ok(())
    }

    fn ensure_vector_add_args(&self, args: &[KernelArg]) -> Result<(vk::Buffer, vk::Buffer, vk::Buffer, u32)> {
        if args.len() != 4 {
            bail!("vector_add expects 4 args: a, b, c, n");
        }
        let a = args[0].as_ptr().ok_or_else(|| anyhow!("arg0 must be pointer"))?;
        let b = args[1].as_ptr().ok_or_else(|| anyhow!("arg1 must be pointer"))?;
        let c = args[2].as_ptr().ok_or_else(|| anyhow!("arg2 must be pointer"))?;
        let n = args[3].as_usize().ok_or_else(|| anyhow!("arg3 must be usize/u32"))?;
        let (abuf, _, _, alen, _, _) = self.get_allocation(a)?;
        let (bbuf, _, _, blen, _, _) = self.get_allocation(b)?;
        let (cbuf, _, _, clen, _, _) = self.get_allocation(c)?;
        let bytes = n.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| anyhow!("byte size overflow"))?;
        if bytes > alen || bytes > blen || bytes > clen {
            bail!("vector_add buffer too small: need {bytes} bytes");
        }
        let n_u32 = u32::try_from(n).context("vector_add n does not fit in u32 push constant")?;
        Ok((abuf, bbuf, cbuf, n_u32))
    }

    fn run_vector_add_spirv(&self, spirv: &[u8], entry: &str, cfg: &LaunchConfig, args: &[KernelArg]) -> Result<()> {
        let (a_buffer, b_buffer, c_buffer, n) = self.ensure_vector_add_args(args)?;
        self.dispatch_spirv(spirv, entry, cfg, &[a_buffer, b_buffer, c_buffer], &n.to_ne_bytes())
    }

    /// `matmul` の引数契約を検証し、Vulkanバッファハンドルと push constant 用の
    /// (m, k, n) を返す。CPU版 `examples/matmul` と同じ行優先(row-major)レイアウトを前提とする:
    /// A は M行K列、B は K行N列、C は M行N列。
    fn ensure_matmul_args(&self, args: &[KernelArg]) -> Result<(vk::Buffer, vk::Buffer, vk::Buffer, u32, u32, u32)> {
        if args.len() != 6 {
            bail!("matmul expects 6 args: a, b, c, m, k, n");
        }
        let a = args[0].as_ptr().ok_or_else(|| anyhow!("arg0 must be pointer"))?;
        let b = args[1].as_ptr().ok_or_else(|| anyhow!("arg1 must be pointer"))?;
        let c = args[2].as_ptr().ok_or_else(|| anyhow!("arg2 must be pointer"))?;
        let m = args[3].as_usize().ok_or_else(|| anyhow!("arg3 (m) must be usize/u32"))?;
        let k = args[4].as_usize().ok_or_else(|| anyhow!("arg4 (k) must be usize/u32"))?;
        let n = args[5].as_usize().ok_or_else(|| anyhow!("arg5 (n) must be usize/u32"))?;

        let (abuf, _, _, alen, _, _) = self.get_allocation(a)?;
        let (bbuf, _, _, blen, _, _) = self.get_allocation(b)?;
        let (cbuf, _, _, clen, _, _) = self.get_allocation(c)?;

        let f32_size = std::mem::size_of::<f32>();
        let a_bytes = m.checked_mul(k).and_then(|v| v.checked_mul(f32_size)).ok_or_else(|| anyhow!("matmul A byte size overflow"))?;
        let b_bytes = k.checked_mul(n).and_then(|v| v.checked_mul(f32_size)).ok_or_else(|| anyhow!("matmul B byte size overflow"))?;
        let c_bytes = m.checked_mul(n).and_then(|v| v.checked_mul(f32_size)).ok_or_else(|| anyhow!("matmul C byte size overflow"))?;
        if a_bytes > alen {
            bail!("matmul buffer A too small: need {a_bytes} bytes, have {alen}");
        }
        if b_bytes > blen {
            bail!("matmul buffer B too small: need {b_bytes} bytes, have {blen}");
        }
        if c_bytes > clen {
            bail!("matmul buffer C too small: need {c_bytes} bytes, have {clen}");
        }

        let m_u32 = u32::try_from(m).context("matmul m does not fit in u32 push constant")?;
        let k_u32 = u32::try_from(k).context("matmul k does not fit in u32 push constant")?;
        let n_u32 = u32::try_from(n).context("matmul n does not fit in u32 push constant")?;
        Ok((abuf, bbuf, cbuf, m_u32, k_u32, n_u32))
    }

    fn run_matmul_spirv(&self, spirv: &[u8], entry: &str, cfg: &LaunchConfig, args: &[KernelArg]) -> Result<()> {
        let (a_buffer, b_buffer, c_buffer, m, k, n) = self.ensure_matmul_args(args)?;
        let mut push = Vec::with_capacity(12);
        push.extend_from_slice(&m.to_ne_bytes());
        push.extend_from_slice(&k.to_ne_bytes());
        push.extend_from_slice(&n.to_ne_bytes());
        self.dispatch_spirv(spirv, entry, cfg, &[a_buffer, b_buffer, c_buffer], &push)
    }

    /// `raid6_xor_parity` の引数契約を検証する(2026-07-30追記、open-raid-zの
    /// NVMe RAID6 ランダムアクセス低速化問題〈parity write penalty〉解決策として
    /// ユーザーから明示指示のあった「open-directx/open-cudaでのハードウェア
    /// アクセラレーター対応」の第一段: RAID6のP-parity(XOR)計算をGPUへ
    /// オフロードする最小実装)。
    ///
    /// レイアウト: `data`バッファは`num_disks`個のブロックを連結したもの
    /// (disk d の word i は `data[d*block_words+i]`)、`parity`バッファは
    /// `block_words`要素。`vector_add`/`matmul`と同じ`args: &[KernelArg]`
    /// 契約に合わせ、ポインタ2つ+usize2つの4引数とする。
    fn ensure_raid6_xor_parity_args(&self, args: &[KernelArg]) -> Result<(vk::Buffer, vk::Buffer, u32, u32)> {
        if args.len() != 4 {
            bail!("raid6_xor_parity expects 4 args: data, parity, num_disks, block_words");
        }
        let data = args[0].as_ptr().ok_or_else(|| anyhow!("arg0 (data) must be pointer"))?;
        let parity = args[1].as_ptr().ok_or_else(|| anyhow!("arg1 (parity) must be pointer"))?;
        let num_disks = args[2].as_usize().ok_or_else(|| anyhow!("arg2 (num_disks) must be usize/u32"))?;
        let block_words = args[3].as_usize().ok_or_else(|| anyhow!("arg3 (block_words) must be usize/u32"))?;

        let (data_buf, _, _, data_len, _, _) = self.get_allocation(data)?;
        let (parity_buf, _, _, parity_len, _, _) = self.get_allocation(parity)?;

        let word_size = std::mem::size_of::<u32>();
        let data_bytes = num_disks
            .checked_mul(block_words)
            .and_then(|v| v.checked_mul(word_size))
            .ok_or_else(|| anyhow!("raid6_xor_parity data byte size overflow"))?;
        let parity_bytes = block_words.checked_mul(word_size).ok_or_else(|| anyhow!("raid6_xor_parity parity byte size overflow"))?;
        if data_bytes > data_len {
            bail!("raid6_xor_parity data buffer too small: need {data_bytes} bytes, have {data_len}");
        }
        if parity_bytes > parity_len {
            bail!("raid6_xor_parity parity buffer too small: need {parity_bytes} bytes, have {parity_len}");
        }

        let num_disks_u32 = u32::try_from(num_disks).context("raid6_xor_parity num_disks does not fit in u32 push constant")?;
        let block_words_u32 = u32::try_from(block_words).context("raid6_xor_parity block_words does not fit in u32 push constant")?;
        Ok((data_buf, parity_buf, num_disks_u32, block_words_u32))
    }

    fn run_raid6_xor_parity_spirv(&self, spirv: &[u8], entry: &str, cfg: &LaunchConfig, args: &[KernelArg]) -> Result<()> {
        let (data_buffer, parity_buffer, num_disks, block_words) = self.ensure_raid6_xor_parity_args(args)?;
        let mut push = Vec::with_capacity(8);
        push.extend_from_slice(&num_disks.to_ne_bytes());
        push.extend_from_slice(&block_words.to_ne_bytes());
        self.dispatch_spirv(spirv, entry, cfg, &[data_buffer, parity_buffer], &push)
    }

    /// SPIR-Vコンピュートシェーダを起動する共通経路。
    ///
    /// `buffers` の各要素は set=0 の連番 binding (STORAGE_BUFFER) に束ねられ、
    /// `push_constants` は non-empty なら stage=COMPUTE のpush constant範囲として渡す。
    /// `vector_add`(push=4byte, buffers=3) と `matmul`(push=12byte, buffers=3) はこの
    /// 一本の経路を共有する。
    fn dispatch_spirv(
        &self,
        spirv: &[u8],
        entry: &str,
        cfg: &LaunchConfig,
        buffers: &[vk::Buffer],
        push_constants: &[u8],
    ) -> Result<()> {
        if !spirv.len().is_multiple_of(4) {
            bail!("SPIR-V byte length must be a multiple of 4");
        }
        let words = bytes_to_u32_words(spirv)?;

        unsafe {
            let shader_info = vk::ShaderModuleCreateInfo::builder().code(&words);
            let shader_module = self.device.create_shader_module(&shader_info, None)
                .context("vkCreateShaderModule failed")?;

            let bindings: Vec<vk::DescriptorSetLayoutBinding> =
                (0..buffers.len() as u32).map(storage_binding).collect();
            let set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
            let set_layout = self.device.create_descriptor_set_layout(&set_layout_info, None)
                .context("vkCreateDescriptorSetLayout failed")?;

            let push_ranges = if push_constants.is_empty() {
                Vec::new()
            } else {
                vec![vk::PushConstantRange::builder()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .size(push_constants.len() as u32)
                    .build()]
            };
            let set_layouts = [set_layout];
            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_ranges);
            let pipeline_layout = self.device.create_pipeline_layout(&pipeline_layout_info, None)
                .context("vkCreatePipelineLayout failed")?;

            let entry_name = CString::new(entry).context("SPIR-V entry contains NUL byte")?;
            let stage = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(&entry_name);
            let pipeline_info = vk::ComputePipelineCreateInfo::builder()
                .stage(stage.build())
                .layout(pipeline_layout);
            let pipeline = self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None)
                .map_err(|(_, e)| anyhow!("vkCreateComputePipelines failed: {e:?}"))?[0];

            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: buffers.len() as u32,
            }];
            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            let descriptor_pool = self.device.create_descriptor_pool(&descriptor_pool_info, None)
                .context("vkCreateDescriptorPool failed")?;
            let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts);
            let descriptor_set = self.device.allocate_descriptor_sets(&alloc_info)
                .context("vkAllocateDescriptorSets failed")?[0];

            let infos: Vec<vk::DescriptorBufferInfo> = buffers
                .iter()
                .map(|&buffer| vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| descriptor_write(descriptor_set, i as u32, std::slice::from_ref(info)))
                .collect();
            self.device.update_descriptor_sets(&writes, &[]);

            let alloc = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = self.device.allocate_command_buffers(&alloc)
                .context("vkAllocateCommandBuffers failed")?[0];
            let begin = vk::CommandBufferBeginInfo::builder();
            self.device.begin_command_buffer(cmd, &begin)
                .context("vkBeginCommandBuffer failed")?;
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            if !push_constants.is_empty() {
                self.device.cmd_push_constants(
                    cmd,
                    pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }
            self.device.cmd_dispatch(cmd, cfg.grid.0, cfg.grid.1, cfg.grid.2);
            self.device.end_command_buffer(cmd).context("vkEndCommandBuffer failed")?;

            let submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&cmd));
            let fence_info = vk::FenceCreateInfo::builder();
            let fence = self.device.create_fence(&fence_info, None).context("vkCreateFence failed")?;
            self.device.queue_submit(self.queue, &[submit.build()], fence)
                .context("vkQueueSubmit failed")?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)
                .context("vkWaitForFences failed")?;

            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &[cmd]);
            self.device.destroy_descriptor_pool(descriptor_pool, None);
            self.device.destroy_pipeline(pipeline, None);
            self.device.destroy_pipeline_layout(pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(set_layout, None);
            self.device.destroy_shader_module(shader_module, None);
        }

        Ok(())
    }
}

impl GpuDevice for VulkanDevice {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        if bytes == 0 {
            return Err(GpuError::OutOfMemory(0).into());
        }
        unsafe {
            let buffer_info = vk::BufferCreateInfo::builder()
                .size(bytes as u64)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self.device.create_buffer(&buffer_info, None)
                .context("vkCreateBuffer failed")?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let (memory_type, coherent) = self.find_host_visible_memory_type(req.memory_type_bits)?;
            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(req.size)
                .memory_type_index(memory_type);
            let memory = self.device.allocate_memory(&alloc_info, None)
                .context("vkAllocateMemory failed")?;
            self.device.bind_buffer_memory(buffer, memory, 0)
                .context("vkBindBufferMemory failed")?;
            // Map the full memory-requirements size, not only the requested byte count.
            // Non-coherent flush/invalidate can then safely use VK_WHOLE_SIZE.
            let mapped = self.device.map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty())
                .context("vkMapMemory failed")? as *mut u8;
            let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
            self.allocations.lock().unwrap().insert(handle, VulkanAllocation {
                buffer,
                memory,
                mapped,
                len: bytes,
                mapped_size: req.size,
                coherent,
            });
            Ok(DevicePtr::new(handle, self.info.id as u32))
        }
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.device_id as usize != self.info.id {
            return Err(GpuError::InvalidPtr(ptr).into());
        }
        let alloc = self.allocations.lock().unwrap().remove(&ptr.addr).ok_or(GpuError::InvalidPtr(ptr))?;
        unsafe {
            self.device.unmap_memory(alloc.memory);
            self.device.destroy_buffer(alloc.buffer, None);
            self.device.free_memory(alloc.memory, None);
        }
        Ok(())
    }

    fn memcpy_h2d(&self, dst: DevicePtr, src: &[u8]) -> Result<()> {
        let (_, memory, mapped, len, memory_size, coherent) = self.get_allocation(dst)?;
        if src.len() > len {
            return Err(GpuError::OutOfMemory(src.len()).into());
        }
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), mapped, src.len()) };
        self.flush_if_needed(memory, memory_size, coherent)
    }

    fn memcpy_d2h(&self, dst: &mut [u8], src: DevicePtr) -> Result<()> {
        let (_, memory, mapped, len, memory_size, coherent) = self.get_allocation(src)?;
        if dst.len() > len {
            return Err(GpuError::InvalidPtr(src).into());
        }
        self.invalidate_if_needed(memory, memory_size, coherent)?;
        unsafe { std::ptr::copy_nonoverlapping(mapped, dst.as_mut_ptr(), dst.len()) };
        Ok(())
    }

    fn memcpy_d2d(&self, dst: DevicePtr, src: DevicePtr, bytes: usize) -> Result<()> {
        let (_, dst_memory, d, dlen, dst_memory_size, dst_coherent) = self.get_allocation(dst)?;
        let (_, src_memory, s, slen, src_memory_size, src_coherent) = self.get_allocation(src)?;
        if bytes > dlen || bytes > slen {
            return Err(GpuError::InvalidPtr(dst).into());
        }
        self.invalidate_if_needed(src_memory, src_memory_size, src_coherent)?;
        unsafe { std::ptr::copy_nonoverlapping(s, d, bytes) };
        self.flush_if_needed(dst_memory, dst_memory_size, dst_coherent)
    }

    fn launch_kernel(&self, kernel: &CompiledKernel, cfg: &LaunchConfig, args: &[KernelArg]) -> Result<()> {
        let spirv = match &kernel.source {
            KernelSource::SpirV(bytes) => bytes,
            other => return Err(GpuError::UnsupportedKernel(other.kind()).into()),
        };
        match kernel.name.as_str() {
            "vector_add" | "vector_add_f32" => self.run_vector_add_spirv(spirv, &kernel.entry, cfg, args),
            "matmul" | "matmul_f32" => self.run_matmul_spirv(spirv, &kernel.entry, cfg, args),
            "raid6_xor_parity" => self.run_raid6_xor_parity_spirv(spirv, &kernel.entry, cfg, args),
            other => bail!(
                "VulkanDevice v0.4.1 only implements vector_add/vector_add_f32, matmul/matmul_f32, and raid6_xor_parity; got `{other}`"
            ),
        }
    }

    fn synchronize(&self) -> Result<()> {
        unsafe { self.device.device_wait_idle().context("vkDeviceWaitIdle failed") }
    }

    fn supports_spirv(&self) -> bool {
        // 実Vulkanデバイス: launch_kernel は KernelSource::SpirV のみを受理する
        // （下記 launch_kernel 実装参照）。ベンダー別スタブ経路(cuBLAS等)より
        // こちらを優先させるための能力フラグ。
        true
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            let handles: Vec<u64> = self.allocations.lock().unwrap().keys().copied().collect();
            for h in handles {
                if let Some(a) = self.allocations.lock().unwrap().remove(&h) {
                    self.device.unmap_memory(a.memory);
                    self.device.destroy_buffer(a.buffer, None);
                    self.device.free_memory(a.memory, None);
                }
            }
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Enumerate a single real Vulkan device for now.
pub fn enumerate_real(start_id: usize) -> Result<Vec<Arc<dyn GpuDevice>>> {
    Ok(vec![VulkanDevice::new(start_id)?])
}

/// Vulkan固有の診断情報（v0.3.6で `vulkan_info` の表示を厚くするために追加）。
#[derive(Clone, Copy)]
pub struct VulkanDiagnostics {
    pub queue_family_index: u32,
    pub device_type: vk::PhysicalDeviceType,
    /// `VK_MAKE_API_VERSION` でエンコードされた生の値。
    pub api_version: u32,
    /// `VK_MAKE_API_VERSION` と同じエンコードだが、意味はベンダー依存
    /// （NVIDIAは major(10)/minor(8)/patch(8)/build(6) の独自レイアウトを使う）。
    pub driver_version: u32,
}

impl std::fmt::Debug for VulkanDiagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDiagnostics")
            .field("queue_family_index", &self.queue_family_index)
            .field("device_type", &self.device_type_str())
            .field("api_version", &self.api_version_str())
            .field("driver_version", &self.driver_version_str())
            .finish()
    }
}

impl VulkanDiagnostics {
    pub fn device_type_str(&self) -> &'static str {
        device_type_str(self.device_type)
    }

    /// `major.minor.patch` 形式。Vulkan標準エンコード（`VK_API_VERSION_*` マクロ相当）。
    pub fn api_version_str(&self) -> String {
        format!(
            "{}.{}.{}",
            vk::api_version_major(self.api_version),
            vk::api_version_minor(self.api_version),
            vk::api_version_patch(self.api_version)
        )
    }

    /// 標準Vulkanエンコードでの `major.minor.patch` に加え、生の値も併記する。
    /// NVIDIAドライバはこのフィールドを独自レイアウトで埋めるため、標準デコードが
    /// 実際のドライババージョン表記（例: 5xx.xx）と一致しない場合がある。
    pub fn driver_version_str(&self) -> String {
        format!(
            "{}.{}.{} (raw=0x{:08x})",
            vk::api_version_major(self.driver_version),
            vk::api_version_minor(self.driver_version),
            vk::api_version_patch(self.driver_version),
            self.driver_version
        )
    }
}

fn device_type_str(t: vk::PhysicalDeviceType) -> &'static str {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => "DISCRETE_GPU",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "INTEGRATED_GPU",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "VIRTUAL_GPU",
        vk::PhysicalDeviceType::CPU => "CPU",
        _ => "OTHER",
    }
}

fn storage_binding(binding: u32) -> vk::DescriptorSetLayoutBinding {
    vk::DescriptorSetLayoutBinding::builder()
        .binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .build()
}

fn descriptor_write(set: vk::DescriptorSet, binding: u32, info: &[vk::DescriptorBufferInfo]) -> vk::WriteDescriptorSet {
    vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(info)
        .build()
}

fn bytes_to_u32_words(bytes: &[u8]) -> Result<Vec<u32>> {
    if !bytes.len().is_multiple_of(4) {
        bail!("SPIR-V length must be multiple of 4");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn vendor_from_id(vendor_id: u32) -> GpuVendor {
    match vendor_id {
        0x10DE => GpuVendor::Nvidia { compute_capability: (0, 0) },
        0x1002 | 0x1022 => GpuVendor::Amd { gfx_version: "unknown".to_string() },
        0x8086 => GpuVendor::Intel { architecture: "unknown".to_string() },
        // 2026-07-25追加: Android/モバイルGPUベンダー(先行するAndroid-Vulkan
        // クロスコンパイル監査が実機Vulkan列挙で未対応と指摘した箇所)。
        // ベンダーIDはpci-ids.ucw.cz/Web検索で裏取り済み(このマシンの
        // 実機はNVIDIA GT 730のみのため、これらの分岐は型チェックのみで
        // 実機Vulkan列挙では未検証)。
        0x5143 => GpuVendor::Qualcomm { architecture: "unknown".to_string() },
        0x13B5 => GpuVendor::Arm { architecture: "unknown".to_string() },
        0x1010 => GpuVendor::ImaginationPowerVr { architecture: "unknown".to_string() },
        _ => GpuVendor::Unknown,
    }
}

fn estimate_device_local_memory(props: &vk::PhysicalDeviceMemoryProperties) -> u64 {
    let mut total = 0u64;
    for i in 0..props.memory_heap_count {
        let heap = props.memory_heaps[i as usize];
        if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            total = total.saturating_add(heap.size);
        }
    }
    total
}
