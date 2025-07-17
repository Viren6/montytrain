use bullet_core::graph::{
    builder::Shape,
    ir::{
        node::AnnotatedNode,
        operation::{GraphIROperation, GraphIROperationCompilable},
        BackendMarker, GraphIR, GraphIRError, GraphIRNodeInfo,
    },
    GraphFunction,
};
use bullet_cuda_backend::{CudaDevice, CudaMarker};

use crate::inputs::{INPUT_SIZE, MAX_MOVES};

#[derive(Debug)]
pub struct ApplyMoveDiff {
    pub weights: AnnotatedNode,
    pub moves: AnnotatedNode,
    pub hl: AnnotatedNode,
}

impl<B: BackendMarker> GraphIROperation<B> for ApplyMoveDiff {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.weights, self.moves, self.hl]
    }

    fn output_shape(&self, _: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        assert_eq!(self.moves.shape.rows(), INPUT_SIZE);
        assert_eq!(self.moves.shape.cols(), MAX_MOVES);
        assert_eq!(self.weights.shape.cols(), INPUT_SIZE);
        assert_eq!(self.hl.shape.cols(), 1);
        assert_eq!(self.weights.shape.rows(), self.hl.shape.rows());

        Ok(Shape::new(self.hl.shape.rows(), MAX_MOVES))
    }
}

impl GraphIROperationCompilable<CudaMarker> for ApplyMoveDiff {
    fn forward_pass(
        &self,
        _node_info: &GraphIRNodeInfo,
        _output_node: usize,
    ) -> GraphFunction<CudaDevice> {
        todo!()
    }

    fn backward_pass(
        &self,
        _node_info: &GraphIRNodeInfo,
        _output_node: usize,
    ) -> GraphFunction<CudaDevice> {
        todo!()
    }
}
