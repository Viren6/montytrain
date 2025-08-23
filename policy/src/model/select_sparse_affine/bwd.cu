extern "C" __global__ void kernel(
    const int num_moves,
    const int batch_size,
    const float* input,
    const float* output_grad,
    const float* weights,
    const int* moves,
    float* input_grad,
    float* weights_grad,
    float* biases_grad
) {
    const int loc_batch = blockIdx.y;
    const int loc_move = blockIdx.x;
    const int move_i = moves[loc_batch * 64 + loc_move];
    if (move_i == -1) {
        return;
    }
    if (threadIdx.x == 0) {
        float grad = output_grad[loc_batch * 64 + loc_move];
        atomicAdd(biases_grad + move_i, grad);
        for (int j = 0; j < 64; j++) {
            int move_j = moves[loc_batch * 64 + j];
            if (move_j != -1) {
                float in_val = input[loc_batch * 64 + j];
                atomicAdd(weights_grad + move_i * num_moves + move_j, grad * in_val);
                float w = weights[move_i * num_moves + move_j];
                atomicAdd(input_grad + loc_batch * 64 + j, grad * w);
            }
        }
    }
}