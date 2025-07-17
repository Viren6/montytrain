use bullet_core::{
    device::OperationError,
    graph::{
        builder::Shape,
        instruction::{GraphInstruction, MaybeUpdateBatchSize},
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
        output_node: usize,
    ) -> GraphFunction<CudaDevice> {
        let weights = NodeId::new(self.weights.idx, NodeIdTy::Values);
        let moves = NodeId::new(self.moves.idx, NodeIdTy::Values);
        let hl = NodeId::new(self.hl.idx, NodeIdTy::Values);
        let output = NodeId::new(output_node, NodeIdTy::Values);

        let mut func = GraphFunction::default();

        func.push(MaybeUpdateBatchSize { input: hl, output });
        func.push(ApplyMoveDiffFwd {
            weights,
            moves,
            hl,
            output,
        });

        func
    }

    fn backward_pass(
        &self,
        _node_info: &GraphIRNodeInfo,
        output_node: usize,
    ) -> GraphFunction<CudaDevice> {
        let moves = NodeId::new(self.moves.idx, NodeIdTy::Values);
        let weights_grad = NodeId::new(self.weights.idx, NodeIdTy::Gradients);
        let hl_grad = NodeId::new(self.hl.idx, NodeIdTy::Gradients);
        let output_grad = NodeId::new(output_node, NodeIdTy::Gradients);

        let mut func = GraphFunction::default();

        func.push(MaybeUpdateBatchSize {
            input: output_grad,
            output: hl_grad,
        });
        func.push(ApplyMoveDiffBwd {
            weights_grad,
            moves,
            hl_grad,
            output_grad,
        });

        func
    }
}

#[derive(Debug)]
pub struct ApplyMoveDiffFwd {
    weights: NodeId,
    moves: NodeId,
    hl: NodeId,
    output: NodeId,
}

impl GraphInstruction<CudaDevice> for ApplyMoveDiffFwd {
    fn execute(&self, graph: &Graph<CudaDevice>) -> Result<(), OperationError<CudaError>> {
        let weights = graph.get(self.weights)?;
        let weights = weights.dense()?;

        let moves = graph.get(self.moves)?;
        let moves = moves.sparse()?;

        let hl = graph.get(self.hl)?;
        let hl = hl.dense()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        let hl_size = hl.single_size();
        let batch_size = hl.batch_size();

        assert_eq!(hl_size % 4, 0);

        if weights.batch_size().is_some()
            || batch_size != moves.batch_size()
            || batch_size != output.batch_size()
        {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if output.single_size() != hl_size * MAX_MOVES {
            return Err(OperationError::InvalidTensorFormat);
        }

        let device = weights.buf.device.clone();

        unsafe {
            let func = device.get_custom_func_or_rtc("apply_move_diff_fwd", || include_str!("diff_fwd.cu").to_string())?;

            let batch_size = batch_size.unwrap_or(1);
            let m4 = hl_size as u32 / 4;
            let threads = m4.min(1024);
            let chunks = m4.div_ceil(threads);

            let grid_dim = (chunks, 64, batch_size as u32);

            let cfg = LaunchConfig { block_dim: (threads, 1, 1), grid_dim, shared_mem_bytes: 0 };

            device
                .stream()
                .launch_builder(&func)
                .arg(&(batch_size as i32))
                .arg(&(hl_size as i32))
                .arg(&weights.buf.buf)
                .arg(&hl.buf.buf)
                .arg(&moves.buf.buf)
                .arg(&mut output.buf.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct ApplyMoveDiffBwd {
    weights_grad: NodeId,
    moves: NodeId,
    hl_grad: NodeId,
    output_grad: NodeId,
}

impl GraphInstruction<CudaDevice> for ApplyMoveDiffBwd {
    fn execute(&self, graph: &Graph<CudaDevice>) -> Result<(), OperationError<CudaError>> {
        let moves = graph.get(self.moves)?;
        let moves = moves.sparse()?;

        let output_grad = graph.get_mut(self.output_grad)?;
        let output_grad = output_grad.dense()?;

        let mut hl_grad = graph.get_mut(self.hl_grad)?;
        let hl_grad = hl_grad.dense_mut()?;

        let mut weights_grad = graph.get_mut(self.weights_grad)?;
        let weights_grad = weights_grad.dense_mut()?;

        let hl_size = hl_grad.single_size();
        let batch_size = hl_grad.batch_size();

        assert_eq!(hl_size % 4, 0);

        if weights_grad.batch_size().is_some()
            || batch_size != moves.batch_size()
            || batch_size != output_grad.batch_size()
        {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if output_grad.single_size() != hl_size * MAX_MOVES {
            return Err(OperationError::InvalidTensorFormat);
        }

        let device = output_grad.buf.device.clone();

        unsafe {
            let func = device.get_custom_func_or_rtc("apply_move_diff_bwd", || include_str!("diff_bwd.cu").to_string())?;

            let batch_size = batch_size.unwrap_or(1);
            let threads = (hl_size as u32).min(1024);
            let chunks = (hl_size as u32).div_ceil(threads);
            let grid_dim = (chunks, 64, batch_size as u32);

            let cfg = LaunchConfig { block_dim: (threads, 1, 1), grid_dim, shared_mem_bytes: 0 };

            device
                .stream()
                .launch_builder(&func)
                .arg(&(batch_size as i32))
                .arg(&(hl_size as i32))
                .arg(&moves.buf.buf)
                .arg(&output_grad.buf.buf)
                .arg(&mut weights_grad.buf.buf)
                .arg(&mut hl_grad.buf.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }
}
