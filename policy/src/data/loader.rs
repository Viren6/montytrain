use bullet_core::{
    graph::builder::Shape,
    trainer::{
        dataloader::{DataLoader, HostDenseMatrix, HostMatrix, HostSparseMatrix, PreparedBatchHost},
        DataLoadingError,
    },
};
use montyformat::chess::Move;

use crate::{
    inputs::{self, INPUT_SIZE, MAX_ACTIVE_BASE, MAX_MOVES},
    moves::{map_move_to_index, NUM_MOVE_INDICES},
};

use super::reader::{DataReader, DecompressedData};

#[derive(Clone)]
pub struct MontyDataLoader {
    reader: DataReader,
    threads: usize,
}

impl MontyDataLoader {
    pub fn new(path: &str, buffer_size_mb: usize, threads: usize) -> Self {
        Self { reader: DataReader::new(path, buffer_size_mb), threads }
    }
}

impl DataLoader for MontyDataLoader {
    type Error = DataLoadingError;

    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(self, batch_size: usize, mut f: F) -> Result<(), Self::Error> {
        self.reader.map_batches(batch_size, |batch| f(prepare(batch, self.threads)));

        Ok(())
    }
}

pub fn prepare(data: &[DecompressedData], threads: usize) -> PreparedBatchHost {
    let batch_size = data.len();
    let chunk_size = batch_size.div_ceil(threads);

    let mut inputs = vec![0; MAX_ACTIVE_BASE * batch_size];
    let mut moves = vec![0; 4 * MAX_MOVES * batch_size];
    let mut dist = vec![0.0; MAX_MOVES * batch_size];
    let mut buckets = vec![0; MAX_MOVES * batch_size];

    std::thread::scope(|s| {
        for ((((data_chunk, input_chunk), moves_chunk), dist_chunk), buckets_chunk) in data
            .chunks(chunk_size)
            .zip(inputs.chunks_mut(MAX_ACTIVE_BASE * chunk_size))
            .zip(moves.chunks_mut(4 * MAX_MOVES * chunk_size))
            .zip(dist.chunks_mut(MAX_MOVES * chunk_size))
            .zip(buckets.chunks_mut(MAX_MOVES * chunk_size))
        {
            s.spawn(move || {
                for (i, point) in data_chunk.iter().enumerate() {
                    let input_offset = MAX_ACTIVE_BASE * i;
                    let moves_offset = 4 * MAX_MOVES * i;
                    let dist_offset = MAX_MOVES * i;

                    let mut j = 0;
                    inputs::map_base_inputs(&point.pos, |feat| {
                        assert!(feat < INPUT_SIZE);
                        input_chunk[input_offset + j] = feat as i32;
                        j += 1;
                    });

                    for k in j..MAX_ACTIVE_BASE {
                        input_chunk[input_offset + k] = -1;
                    }

                    assert!(j <= MAX_ACTIVE_BASE, "More inputs provided than the specified maximum!");

                    let mut total = 0;
                    let mut distinct = 0;

                    for &(mov, visits) in &point.moves[..point.num] {
                        total += visits;

                        let mov = Move::from(mov);
                        let diff = inputs::get_diff(&point.pos, &point.castling, mov);

                        for k in 0..4 {
                            moves_chunk[moves_offset + 4 * distinct + k] = diff[k];
                        }

                        dist_chunk[dist_offset + distinct] = f32::from(visits);
                        buckets_chunk[dist_offset + distinct] = map_move_to_index(&point.pos, mov) as i32;
                        distinct += 1;
                    }

                    for k in 4 * distinct..4 * MAX_MOVES {
                        moves_chunk[moves_offset + k] = -1;
                    }

                    for k in distinct..MAX_MOVES {
                        buckets_chunk[dist_offset + k] = -1;
                    }

                    let total = f32::from(total);

                    for idx in 0..distinct {
                        dist_chunk[dist_offset + idx] /= total;
                    }
                }
            });
        }
    });

    let mut prep = PreparedBatchHost { batch_size, inputs: Default::default() };

    unsafe {
        prep.inputs.insert(
            "inputs".to_string(),
            HostMatrix::Sparse(HostSparseMatrix::new(inputs, batch_size, Shape::new(INPUT_SIZE, 1), MAX_ACTIVE_BASE)),
        );

        prep.inputs.insert(
            "moves".to_string(),
            HostMatrix::Sparse(HostSparseMatrix::new(
                moves,
                batch_size,
                Shape::new(INPUT_SIZE, MAX_MOVES),
                4 * MAX_MOVES,
            )),
        );

        prep.inputs.insert(
            "buckets".to_string(),
            HostMatrix::Sparse(HostSparseMatrix::new(
                buckets,
                batch_size,
                Shape::new(NUM_MOVE_INDICES, MAX_MOVES),
                MAX_MOVES,
            )),
        );
    }

    prep.inputs.insert(
        "targets".to_string(),
        HostMatrix::Dense(HostDenseMatrix::new(dist, batch_size, Shape::new(MAX_MOVES, 1))),
    );

    prep
}
