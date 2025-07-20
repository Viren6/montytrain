pub mod data;
pub mod inputs;
pub mod model;
pub mod moves;

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

    let end_superbatch = 2400;
    let initial_lr = 0.001;
    let final_lr = 0.00001;

    let steps = TrainingSteps { batch_size: 4096, batches_per_superbatch: 6104, start_superbatch: 1, end_superbatch };

    let schedule = TrainingSchedule {
        steps,
        log_rate: 64,
        lr_schedule: Box::new(|_, sb| {
            if sb >= end_superbatch {
                return final_lr;
            }

            let lambda = sb as f32 / end_superbatch as f32;
            initial_lr * (final_lr / initial_lr).powf(lambda)
        }),
    };

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
