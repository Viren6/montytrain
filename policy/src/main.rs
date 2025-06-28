mod inputs;
mod loader;
mod moves;
mod preparer;
mod trainer;

use bullet::{
    nn::{
        optimiser::{AdamWParams, Optimiser},
        Activation, ExecutionContext, Graph, InitSettings, NetworkBuilder, Shape,
    },
    trainer::{
        logger,
        save::{Layout, QuantTarget, SavedFormat},
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
        NetworkTrainer,
    },
};

use trainer::Trainer;

const ID: &str = "policy001";

fn main() {
    //let data_preparer = preparer::DataPreparer::new("data/policygen6.binpack", 4096);
    let data_preparer = preparer::DataPreparer::new(
        "/home/privateclient/monty_value_training/interleaved.binpack",
        96000,
    );

    let size = 12288;

    let graph = network(size);

    let optimiser_params = AdamWParams {
        decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        min_weight: -0.99,
        max_weight: 0.99,
    };

    let mut trainer = Trainer {
        optimiser: Optimiser::new(graph, optimiser_params).unwrap(),
    };

    let schedule = TrainingSchedule {
        net_id: ID.to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 600,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::ExponentialDecayLR {
            initial_lr: 0.001,
            final_lr: 0.00001,
            final_superbatch: 600,
        },
        save_rate: 40,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 32,
    };

    logger::clear_colours();
    println!("{}", logger::ansi("Beginning Training", "34;1"));
    schedule.display();
    settings.display();

    trainer.train_custom(
        &data_preparer,
        &Option::<preparer::DataPreparer>::None,
        &schedule,
        &settings,
        |sb, trainer, schedule, _| {
            if schedule.should_save(sb) {
                trainer
                    .save_weights_portion(
                        &format!("checkpoints/{ID}-{sb}.network"),
                        &[
                            SavedFormat::new("l0w", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("l0b", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new(
                                "l1w",
                                QuantTarget::Float,
                                Layout::Transposed(Shape::new(moves::NUM_MOVES, size / 2)),
                            ),
                            SavedFormat::new("l1b", QuantTarget::Float, Layout::Normal),
                        ],
                    )
                    .unwrap();
            }
        },
    );
}

fn network(size: usize) -> Graph {
    let builder = NetworkBuilder::default();
    let b = &builder;

    let inputs = b.new_sparse_input(
        "inputs",
        Shape::new(inputs::INPUT_SIZE, 1),
        inputs::MAX_ACTIVE,
    );
    let mask = b.new_sparse_input("mask", Shape::new(moves::NUM_MOVES, 1), moves::MAX_MOVES);
    let dist = b.new_dense_input("dist", Shape::new(moves::MAX_MOVES, 1));

    let see_val = b.new_dense_input(
        // raw SEE scores per move
        "see_val",
        Shape::new(moves::NUM_MOVES, 1),
    );

    /* ---------- global learnable scale  (starts at 0.002) ---------- */
    let alpha = b.new_weights(
        "see_alpha",
        Shape::new(1, 1),
        InitSettings::Normal {
            mean: 0.002,
            stdev: 0.0,
        },
    );

    /* ---------- apply tanh to SEE values -------- */

    // scaled = see_val * alpha
    let scaled = see_val.matmul(alpha);

    // tanh via 2sigmoid(2x)-1 because Bullet has no built-in Tanh
    let two = b.new_weights(
        "see_two",
        Shape::new(1, 1),
        InitSettings::Normal {
            mean: 2.0,
            stdev: 0.0,
        },
    );
    let neg_one = b.new_weights(
        "see_neg_one",
        Shape::new(moves::NUM_MOVES, 1),
        InitSettings::Normal {
            mean: -1.0,
            stdev: 0.0,
        },
    );

    let tanh_vals = scaled
        .matmul(two)                  // 2x
        .activate(Activation::Sigmoid)
        .matmul(two)                  // 2sigmoid(2x)
        + neg_one; // -1

    let see_vec = tanh_vals;
    /* --------------------------------------------------------------- */

    let l0 = builder.new_affine("l0", inputs::INPUT_SIZE, size);
    let l1 = builder.new_affine("l1", size / 2, moves::NUM_MOVES);

    // branch layers
    let l3 = builder.new_affine("l3", 256, moves::NUM_MOVES); // 256  1 880

    let mut trunk = l0.forward(inputs).activate(Activation::CReLU);
    trunk = trunk.pairwise_mul(); // 12 288  **6 144**

    // -- main path (unchanged)
    let main = l1.forward(trunk.clone()); // 6 144  1 880

    // -- secondary path
    let mut branch = trunk;
    for _ in 0..5 {
        branch = branch.pairwise_mul();
    } // 6 144  **192**

    branch = branch.concat(see_vec); // 192 + 1 880 = **2 072**

    let l2_in = (size / 64) + moves::NUM_MOVES; // 192 + 1 880
    let l2 = builder.new_affine("l2", l2_in, 256); // 2 072  256

    branch = l2.forward(branch).activate(Activation::CReLU); // 2072  256
    branch = l3.forward(branch); // 256  1 880

    // combine paths and apply loss
    let out = main + branch; // element-wise sum
    out.masked_softmax_crossentropy_loss(dist, mask);

    builder.build(ExecutionContext::default())
}
