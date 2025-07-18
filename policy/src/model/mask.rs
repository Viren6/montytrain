use bullet_core::{
    device::OperationError,
    graph::{
        builder::Shape,
        instruction::{GraphInstruction, LinearCombination, MaybeUpdateBatchSize},
        ir::{
            node::AnnotatedNode,
            operation::{GraphIROperation, GraphIROperationCompilable},
            BackendMarker, GraphIR, GraphIRError, GraphIRNodeInfo,
        },
        Graph, GraphFunction, NodeId, NodeIdTy,
    },
};
use bullet_cuda_backend::{CudaDevice, CudaError, CudaMarker};
use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::inputs::MAX_MOVES;

#[derive(Debug)]
pub struct MaskOutNonMoves {
    pub input: AnnotatedNode,
    pub moves: AnnotatedNode,
}

impl<B: BackendMarker> GraphIROperation<B> for MaskOutNonMoves {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input, self.moves]
    }

    fn output_shape(&self, _: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        assert_eq!(self.input.shape.size(), MAX_MOVES);

        Ok(self.input.shape)
    }
}

impl GraphIROperationCompilable<CudaMarker> for MaskOutNonMoves {
    fn forward_pass(&self, _node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<CudaDevice> {
        let moves = NodeId::new(self.moves.idx, NodeIdTy::Values);
        let input = NodeId::new(self.input.idx, NodeIdTy::Values);
        let output = NodeId::new(output_node, NodeIdTy::Values);

        let mut func = GraphFunction::default();

        func.push(MaybeUpdateBatchSize { input, output });
        func.push(MaskOutNonMovesFwd { input, moves, output });

        func
    }

    fn backward_pass(&self, _node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<CudaDevice> {
        let output_grad = NodeId::new(output_node, NodeIdTy::Gradients);
        let input_grad = NodeId::new(self.input.idx, NodeIdTy::Gradients);

        let mut func = GraphFunction::default();

        func.push(MaybeUpdateBatchSize { input: output_grad, output: input_grad });
        func.push(LinearCombination { input: output_grad, input_mul: 1.0, output: input_grad, output_mul: 1.0 });

        func
    }
}

#[derive(Debug)]
pub struct MaskOutNonMovesFwd {
    input: NodeId,
    moves: NodeId,
    output: NodeId,
}

impl GraphInstruction<CudaDevice> for MaskOutNonMovesFwd {
    fn execute(&self, graph: &Graph<CudaDevice>) -> Result<(), OperationError<CudaError>> {
        let input = graph.get(self.input)?;
        let input = input.dense()?;

        let moves = graph.get(self.moves)?;
        let moves = moves.sparse()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        let single_size = input.single_size();
        let batch_size = input.batch_size();

        assert_eq!(single_size, MAX_MOVES);

        if batch_size != moves.batch_size() || batch_size != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if single_size != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let device = input.buf.device.clone();

        unsafe {
            let func = device.get_custom_func_or_rtc("mask_out", || include_str!("mask.cu").to_string())?;

            let batch_size = batch_size.unwrap_or(1);
            let threads = 1024;
            let dim = (single_size * batch_size).div_ceil(threads);

            let cfg =
                LaunchConfig { block_dim: (threads as u32, 1, 1), grid_dim: (dim as u32, 1, 1), shared_mem_bytes: 0 };

            device
                .stream()
                .launch_builder(&func)
                .arg(&(batch_size as i32))
                .arg(&(single_size as i32))
                .arg(&moves.buf.buf)
                .arg(&input.buf.buf)
                .arg(&mut output.buf.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }
}
