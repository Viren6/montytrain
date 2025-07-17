pub mod data;
pub mod inputs;
pub mod model;

use bullet_core::{
    device::Device,
    optimiser::{
        adam::{AdamW, AdamWParams},
        Optimiser,
    },
    trainer::{
        schedule::{TrainingSchedule, TrainingSteps},
        Trainer,
    },
};
use bullet_cuda_backend::CudaDevice;

use crate::data::MontyDataLoader;

fn main() {
    let device = CudaDevice::new(0).unwrap();

    let graph = model::make(device, 1024);

    let mut trainer = Trainer {
        optimiser: Optimiser::<_, AdamW<_>>::new(graph, AdamWParams::default()).unwrap(),
        state: (),
    };

    let dataloader = MontyDataLoader::new("data/policygen6.binpack", 1024, 4);

    let schedule = TrainingSchedule {
        out_dir: "checkpoints".to_string(),
        steps: TrainingSteps {
            batch_size: 4096,
            batches_per_superbatch: 256,
            start_superbatch: 1,
            end_superbatch: 100,
        },
        net_id: "policy".to_string(),
        save_rate: 10,
        log_rate: 16,
        lr_schedule: Box::new(|_, _| 0.001),
    };

    trainer
        .train_custom(schedule, dataloader, |_, _, _, _| {}, |_, _| {})
        .unwrap();
}
