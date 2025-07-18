extern "C" __global__ void kernel(
    const int batch_size,
    const int* moves,
    float* output
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= 64 * batch_size) return;

    const int4 move = reinterpret_cast<const int4*>(moves)[tid];
    output[tid] = move.x != -1 ? 0.0F : -10000.0F;
}