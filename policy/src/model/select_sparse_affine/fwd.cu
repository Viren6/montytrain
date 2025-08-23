extern "C" __global__ void kernel(
    const int num_moves,
    const int batch_size,
    const float* weights,
    const float* biases,
    const float* input,
    const int* moves,
    float* output
) {
    const int loc_batch = blockIdx.y;
    const int loc_move = blockIdx.x;
    const int move_i = moves[loc_batch * 64 + loc_move];
    if (move_i == -1) {
        if (threadIdx.x == 0) {
            output[loc_batch * 64 + loc_move] = -10000.0f;
        }
        return;
    }
    if (threadIdx.x == 0) {
        float val = biases[move_i];
        for (int j = 0; j < 64; j++) {
            int move_j = moves[loc_batch * 64 + j];
            if (move_j != -1) {
                val += weights[move_i * num_moves + move_j] * input[loc_batch * 64 + j];
            }
        }
        output[loc_batch * 64 + loc_move] = val;
    }
}