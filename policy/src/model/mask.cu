extern "C" __global__ void kernel(
    const int batch_size,
    const int single_size,
    const int* moves,
    const float* input,
    float* output
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int loc_in_batch = tid / single_size;
    const int loc_in_moves = tid % single_size;

    if (loc_in_batch >= batch_size || loc_in_moves >= single_size)
    {
        return;
    }

    const int4 move = reinterpret_cast<const int4*>(moves)[64 * loc_in_batch + loc_in_moves];

    output[tid] = move.x != -1 ? input[tid] : -10000.0F;
}