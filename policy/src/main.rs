pub mod data;
pub mod inputs;
pub mod model;

use bullet_core::device::Device;
use bullet_cuda_backend::CudaDevice;

fn main() {
    let device = CudaDevice::new(0).unwrap();

    let graph = model::make(device, 1024);

    graph.display("forward").unwrap();
}
