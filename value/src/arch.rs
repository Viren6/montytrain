use bullet::{
    inputs::InputType,
    operations,
    optimiser::{AdamWOptimiser, AdamWParams},
    outputs, Activation, ExecutionContext, Graph, GraphBuilder, Node, QuantTarget, Shape, Trainer,
};

use crate::input::ThreatInputs;

pub fn make_trainer(l1: usize) -> Trainer<AdamWOptimiser, ThreatInputs, outputs::Single> {
    let num_inputs = ThreatInputs.size();

    let (mut graph, output_node) = build_network(num_inputs, l1);

    let sizes = [num_inputs, l1];

    for (i, &size) in sizes.iter().enumerate() {
        graph
            .get_weights_mut(&format!("l{i}w"))
            .seed_random(0.0, 1.0 / (size as f32).sqrt(), true);

        graph
            .get_weights_mut(&format!("l{i}b"))
            .seed_random(0.0, 1.0 / (size as f32).sqrt(), true);
    }

    Trainer::new(
        graph,
        output_node,
        AdamWParams::default(),
        ThreatInputs,
        outputs::Single,
        vec![
            ("l0w".to_string(), QuantTarget::Float),
            ("l0b".to_string(), QuantTarget::Float),
            ("l1w".to_string(), QuantTarget::Float),
            ("l1b".to_string(), QuantTarget::Float),
        ],
    )
}

fn build_network(inputs: usize, l1: usize) -> (Graph, Node) {
    let mut builder = GraphBuilder::default();

    // inputs
    let stm = builder.create_input("stm", Shape::new(inputs, 1));
    let targets = builder.create_input("targets", Shape::new(3, 1));

    // trainable weights
    let l0w = builder.create_weights("l0w", Shape::new(l1, inputs));
    let l0b = builder.create_weights("l0b", Shape::new(l1, 1));
    let l1w = builder.create_weights("l1w", Shape::new(3, l1));
    let l1b = builder.create_weights("l1b", Shape::new(3, 1));

    // inference
    let l1 = operations::affine(&mut builder, l0w, stm, l0b);
    let l1 = operations::activate(&mut builder, l1, Activation::SCReLU);

    let predicted = operations::affine(&mut builder, l1w, l1, l1b);
    operations::softmax_crossentropy_loss(&mut builder, predicted, targets);

    // graph, output node
    (builder.build(ExecutionContext::default()), predicted)
}
