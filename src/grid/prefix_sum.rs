use nalgebra::DVector;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::GpuVector;
use wgcore::Shader;
use wgpu::{BufferUsages, ComputePipeline, Device};

/// This is a special variant of the prefix sum algorithm that assumes a 0 is always prepended
/// as the first element in the vector.

#[derive(Shader)]
#[shader(src = "prefix_sum.wgsl", composable = false)]
pub struct WgPrefixSum {
    prefix_sum: ComputePipeline,
    add_data_grp: ComputePipeline,
}

impl WgPrefixSum {
    const THREADS: u32 = 256;

    pub fn queue<'a>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        workspace: &mut PrefixSumWorkspace,
        data: &GpuVector<u32>,
    ) {
        // If this assert fails, the kernel launches bellow must be changed because we are using
        // a fixed size for the shared memory currently.
        assert_eq!(
            Self::THREADS,
            256,
            "Internal error: prefix sum assumes a thread count equal to 256"
        );

        workspace.reserve(queue.device(), data.len() as u32);

        let ngroups0 = workspace.stages[0].buffer.len() as u32;
        let aux0 = &workspace.stages[0].buffer;
        KernelInvocationBuilder::new(queue, &self.prefix_sum)
            .bind0([data.buffer(), aux0.buffer()])
            .queue(ngroups0);

        for i in 0..workspace.num_stages - 1 {
            let ngroups = workspace.stages[i + 1].buffer.len() as u32;
            let buf = workspace.stages[i].buffer.buffer();
            let aux = workspace.stages[i + 1].buffer.buffer();

            KernelInvocationBuilder::new(queue, &self.prefix_sum)
                .bind0([buf, aux])
                .queue(ngroups);
        }

        if workspace.num_stages > 2 {
            for i in (0..workspace.num_stages - 2).rev() {
                let ngroups = workspace.stages[i + 1].buffer.len() as u32;
                let buf = workspace.stages[i].buffer.buffer();
                let aux = workspace.stages[i + 1].buffer.buffer();

                KernelInvocationBuilder::new(queue, &self.add_data_grp)
                    .bind0([buf, aux])
                    .queue(ngroups);
            }
        }

        if workspace.num_stages > 1 {
            KernelInvocationBuilder::new(queue, &self.add_data_grp)
                .bind0([data.buffer(), aux0.buffer()])
                .queue(ngroups0);
        }
    }

    pub fn eval_cpu(&self, v: &mut DVector<u32>) {
        for i in 0..v.len() - 1 {
            v[i + 1] += v[i];
        }

        // NOTE: we actually have a special variant of the prefix-sum
        //       where the result is as if a 0 was appendend to the input vector.
        for i in (1..v.len()).rev() {
            v[i] = v[i - 1];
        }

        v[0] = 0;
    }
}

struct PrefixSumStage {
    capacity: u32,
    buffer: GpuVector<u32>,
}

#[derive(Default)]
pub struct PrefixSumWorkspace {
    stages: Vec<PrefixSumStage>,
    num_stages: usize,
}

impl PrefixSumWorkspace {
    pub fn new() -> Self {
        Self {
            stages: vec![],
            num_stages: 0,
        }
    }

    pub fn with_capacity(device: &Device, buffer_len: u32) -> Self {
        let mut result = Self {
            stages: vec![],
            num_stages: 0,
        };
        result.reserve(device, buffer_len);
        result
    }

    pub fn reserve(&mut self, device: &Device, buffer_len: u32) {
        let mut stage_len = buffer_len.div_ceil(WgPrefixSum::THREADS);

        if self.stages.is_empty() || self.stages[0].capacity < stage_len {
            // Reinitialize the auxiliary buffers.
            self.stages.clear();

            while stage_len != 1 {
                let buffer = GpuVector::init(
                    device,
                    DVector::<u32>::zeros(stage_len as usize),
                    BufferUsages::STORAGE,
                );
                self.stages.push(PrefixSumStage {
                    capacity: stage_len,
                    buffer,
                });

                stage_len = stage_len.div_ceil(WgPrefixSum::THREADS);
            }

            // The last stage always has only 1 element.
            self.stages.push(PrefixSumStage {
                capacity: 1,
                buffer: GpuVector::init(device, DVector::<u32>::zeros(1), BufferUsages::STORAGE),
            });
            self.num_stages = self.stages.len();
        } else if self.stages[0].buffer.len() as u32 != stage_len {
            // The stages have big enough buffers, but we need to adjust their length.
            self.num_stages = 0;
            while stage_len != 1 {
                self.num_stages += 1;
                stage_len = stage_len.div_ceil(WgPrefixSum::THREADS);
            }

            // The last stage always has only 1 element.
            self.num_stages += 1;
        }
    }

    /*
    pub fn read_max_scan_value(&mut self) -> cust::error::CudaResult<u32> {
        for stage in &self.stages {
            if stage.len == 1 {
                // This is the last stage, it contains the total sum.
                let mut value = [0u32];
                stage.buffer.index(0).copy_to(&mut value)?;
                return Ok(value[0]);
            }
        }

        panic!("The GPU prefix sum has not been initialized yet.")
    }
    */
}

#[cfg(test)]
mod test {
    use super::{PrefixSumWorkspace, WgPrefixSum};
    use nalgebra::{DMatrix, DVector};
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::KernelInvocationQueue;
    use wgcore::tensor::{GpuVector, TensorBuilder};
    use wgcore::Shader;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_prefix_sum() {
        const LEN: u32 = 15071;

        let gpu = GpuInstance::new().await.unwrap();
        let prefix_sum = WgPrefixSum::from_device(gpu.device()).unwrap();
        let mut queue = KernelInvocationQueue::new(gpu.device());

        let inputs = vec![
            DVector::<u32>::from_fn(LEN as usize, |i, _| 1),
            DVector::<u32>::from_fn(LEN as usize, |i, _| i as u32),
            DVector::<u32>::new_random(LEN as usize).map(|e| e % 10_000),
        ];

        for v_cpu in inputs {
            let mut encoder = gpu.device().create_command_encoder(&Default::default());
            let v_gpu = GpuVector::init(
                gpu.device(),
                &v_cpu,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            );
            let staging = GpuVector::uninit(
                gpu.device(),
                v_cpu.len() as u32,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            );

            let mut workspace = PrefixSumWorkspace::with_capacity(gpu.device(), v_cpu.len() as u32);
            prefix_sum.queue(&mut queue, &mut workspace, &v_gpu);

            queue.encode(&mut encoder, None);
            staging.copy_from(&mut encoder, &v_gpu);

            let t0 = std::time::Instant::now();
            gpu.queue().submit(Some(encoder.finish()));
            let gpu_result = staging.read(gpu.device()).await.unwrap();
            println!("Gpu time: {}", t0.elapsed().as_secs_f32());

            let mut cpu_result = v_cpu.clone();

            let t0 = std::time::Instant::now();
            prefix_sum.eval_cpu(&mut cpu_result);
            println!("Cpu time: {}", t0.elapsed().as_secs_f32());
            // println!("input: {:?}", v_cpu);
            // println!("cpu output: {:?}", cpu_result);
            // println!("gpu output: {:?}", gpu_result);

            assert_eq!(DVector::from(gpu_result), cpu_result);
        }
    }
}
