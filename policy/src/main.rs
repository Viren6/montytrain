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

    let (graph, node) = model::make(device, 512);

    let params = AdamWParams { min_weight: -0.99, max_weight: 0.99, ..Default::default() };
    let optimiser = Optimiser::<_, AdamW<_>>::new(graph, params).unwrap();

    let mut trainer = Trainer { optimiser, state: () };

    let dataloader = MontyDataLoader::new("data/policygen6.binpack", 1024, 4);

    let steps =
        TrainingSteps { batch_size: 4096, batches_per_superbatch: 256, start_superbatch: 1, end_superbatch: 10 };

    let schedule = TrainingSchedule { steps, log_rate: 16, lr_schedule: Box::new(|_, _| 0.001) };

    trainer
        .train_custom(
            schedule,
            dataloader,
            |_, _, _, _| {},
            |trainer, superbatch| {
                if superbatch % 10 == 0 || superbatch == steps.end_superbatch {
                    println!("Saving Checkpoint");
                    let dir = format!("checkpoints/policy-{superbatch}");
                    let _ = std::fs::create_dir(&dir);
                    trainer.optimiser.write_to_checkpoint(&dir).unwrap();
                    model::save_quantised(&trainer.optimiser.graph, &format!("{dir}/quantised.bin")).unwrap();
                }
            },
        )
        .unwrap();

    model::eval(&mut trainer.optimiser.graph, node, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}
