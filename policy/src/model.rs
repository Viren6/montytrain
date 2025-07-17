mod diff;
mod mask;

use bullet_core::graph::{
    builder::{GraphBuilder, Shape},
    Graph,
};
use bullet_cuda_backend::CudaDevice;

use crate::{
    inputs::{INPUT_SIZE, MAX_ACTIVE_BASE, MAX_MOVES},
    model::mask::MaskOutNonMoves,
};

pub fn make(device: CudaDevice, hl: usize) -> Graph<CudaDevice> {
    let builder = GraphBuilder::default();

    let inputs = builder.new_sparse_input("inputs", Shape::new(INPUT_SIZE, 1), MAX_ACTIVE_BASE);
    let moves = builder.new_sparse_input("moves", Shape::new(INPUT_SIZE, MAX_MOVES), 4 * MAX_MOVES);
    let targets = builder.new_dense_input("targets", Shape::new(1, MAX_MOVES));

    let l0 = builder.new_affine("l0", INPUT_SIZE, hl);
    let l1 = builder.new_affine("l1", hl, 1);

    let base_hl = l0.forward(inputs);
    let move_hls = builder
        .apply(diff::ApplyMoveDiff {
            weights: l0.weights.annotated_node(),
            moves: moves.annotated_node(),
            hl: base_hl.annotated_node(),
        })
        .screlu();

    let ones = builder.new_constant(Shape::new(1, MAX_MOVES), &[1.0; MAX_MOVES]);
    let logits = l1.weights.matmul(move_hls) + l1.bias.matmul(ones);
    let masked = builder.apply(MaskOutNonMoves {
        input: logits.annotated_node(),
        moves: moves.annotated_node(),
    });
    let loss = masked
        .softmax_crossentropy_loss(targets)
        .reshape(Shape::new(MAX_MOVES, 1));
    let _ = ones.matmul(loss);

    builder.build(device)
}
