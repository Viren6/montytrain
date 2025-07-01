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

    /* ---------- raw SEE scores per move (column) ---------- *
    * later code needs (NUM_MOVES, 1)                       */
    let see_val_col = b.new_dense_input(
        "see_val",
        Shape::new(moves::NUM_MOVES, 1),
    );

    /* ---------- create a ROW view for safe scalar-left math ---------- *
    * (1, NUM_MOVES) is still a single-batch tensor in Bullet          */
    let see_val = see_val_col.reshape(Shape::new(1, moves::NUM_MOVES));

    /* ---------- learnable & constant scalars (all 1 x 1) ---------- */
    let alpha = b.new_weights(
        "see_alpha",
        Shape::new(1, 1),
        InitSettings::Normal { mean: 0.002, stdev: 0.0 },
    );
    let two = b.new_weights(
        "see_two",
        Shape::new(1, 1),
        InitSettings::Normal { mean: 2.0, stdev: 0.0 },
    );

    /* ---------- row vector of -1s to finish tanh(x) = 2a(2x) - 1 ---------- */
    let neg_one = b.new_weights(
        "see_neg_one",
        Shape::new(1, moves::NUM_MOVES),
        InitSettings::Normal { mean: -1.0, stdev: 0.0 },
    );

    /* ---------- 1.  a x SEE  (1x1  x  1xN = 1xN) ---------- */
    let scaled   = alpha.matmul(see_val);

    /* ---------- 2.  2x      ---------- */
    let two_x    = two.matmul(scaled);

    /* ---------- 3.  a(2x)   ---------- */
    let sig      = two_x.activate(Activation::Sigmoid);

    /* ---------- 4.  2a(2x) ---------- */
    let two_sig  = two.matmul(sig);

    /* ---------- 5.  tanh(x) = 2a(2x) - 1  ---------- */
    let tanh_row = two_sig + neg_one;                 // shape: (1, NUM_MOVES)

    /* ---------- 6.  reshape back to column for downstream code ---------- */
    let tanh_vals = tanh_row.reshape(Shape::new(moves::NUM_MOVES, 1));  // (NUM_MOVES,1)


    let see_vec = tanh_vals;
    /* --------------------------------------------------------------- */

    let l0 = builder.new_affine("l0", inputs::INPUT_SIZE, size);
    let l1 = builder.new_affine("l1", size / 2, moves::NUM_MOVES);

    // branch layers
    let l2 = builder.new_affine("l2", size / 4, moves::NUM_MOVES); // 3 072 -> 1 880

    let mut trunk = l0.forward(inputs).activate(Activation::CReLU);
    trunk = trunk.pairwise_mul(); // 12 288  **6 144**

    // -- main path (unchanged)
    let main = l1.forward(trunk.clone()); // 6 144  1 880

    // -- secondary path ---------------------------------------------------
    let branch_in = trunk.pairwise_mul();               // 6 144 → 3 072   (features)
    let mut branch = l2.forward(branch_in.clone());     // 3 072 → 1 880   (W·x)

    // ---- add the SEE term to the weights (W  +  see_j) ------------------
    // 1.  compute  Σ xᵢ   once for the current position in the batch
    let ones = builder.new_weights(
        "ones_row",
        Shape::new(1, size / 4),                        // 1 × 3 072
        InitSettings::Normal { mean: 1.0, stdev: 0.0 }, // constant “1”
    );
    let inp_sum = ones.matmul(branch_in);               // (1 × 1)  ==  Σ xᵢ

    // 2.  broadcast it and scale by tanh‑SEE for every move
    let see_term = see_vec.matmul(inp_sum);             // (1 880 × 1)

    // 3.  add to the affine output →  (W x  +  see_j·Σ xᵢ)
    branch = branch + see_term;                         // 1 880 × 1
    // ---------------------------------------------------------------------

    // combine paths and apply loss
    let out = main + branch;
    out.masked_softmax_crossentropy_loss(dist, mask);

    builder.build(ExecutionContext::default())
}
