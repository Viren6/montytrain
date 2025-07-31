extern "C" __global__ void kernel(
    const int in_size,
    const int batch_size,
    const float* weights,
    const float* input,
    const int* moves,
    const float* output_grad,
    float* input_grad,
    float* weights_grad,
    float* biases_grad
) {

    const int loc_in_batch = blockIdx.y;
    const int loc_in_moves = blockIdx.x;
    const int tid = threadIdx.x;
    const int locmb = loc_in_batch * 64 + loc_in_moves;
    const int move = moves[locmb];
    
    if (move != -1)
    {
        const float grd = output_grad[locmb];

        const float4* tW = reinterpret_cast<const float4*>(weights + in_size * move);
        const float4* tI = reinterpret_cast<const float4*>(input + in_size * loc_in_batch);

        if (tid == 0) atomicAdd(biases_grad + move, grd);

        for (int idx = tid; idx < in_size / 4; idx += blockDim.x)
        {
            const float4 ti = tI[idx];
            const float4 tw = tW[idx];

            int wg_base = in_size * move + idx * 4;
            atomicAdd(weights_grad + wg_base    , grd * ti.x);
            atomicAdd(weights_grad + wg_base + 1, grd * ti.y);
            atomicAdd(weights_grad + wg_base + 2, grd * ti.z);
            atomicAdd(weights_grad + wg_base + 3, grd * ti.w);

            int ig_base = in_size * loc_in_batch + idx * 4;
            atomicAdd(input_grad + ig_base    , grd * tw.x);
            atomicAdd(input_grad + ig_base + 1, grd * tw.y);
            atomicAdd(input_grad + ig_base + 2, grd * tw.z);
            atomicAdd(input_grad + ig_base + 3, grd * tw.w);		
        }
    }
}