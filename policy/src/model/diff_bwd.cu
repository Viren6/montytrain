extern "C" __global__ void kernel(
    const int batch_size,
    const int hl_size,
    const int* moves,
    const float* ogrd,
    const float* out,
    float* wgrd,
    float* hgrd
) {
    const int loc_in_batch = blockIdx.z;
    const int loc_in_moves = blockIdx.y;
    const int loc_in_neurons = blockIdx.x * blockDim.x + threadIdx.x;

    if (loc_in_neurons >= hl_size || loc_in_batch >= batch_size || loc_in_moves >= 64)
    {
        return;
    }

    const int locmb = 64 * loc_in_batch + loc_in_moves;
    const int4 move = reinterpret_cast<const int4*>(moves)[locmb];

    if (move.x != -1) {
        const int plc = hl_size * locmb + loc_in_neurons;
        const float tout = out[plc];
        const float grd = tout > 0.0F && tout < 1.0F ? 2.0F * sqrtf(tout) * ogrd[plc] : 0.0F;

        atomicAdd(hgrd + hl_size * loc_in_batch + loc_in_neurons, grd);
        atomicAdd(wgrd + hl_size * move.x + loc_in_neurons, -grd);

        if (move.y != -1)
        {
            atomicAdd(wgrd + hl_size * move.y + loc_in_neurons, -grd);
        }

        atomicAdd(wgrd + hl_size * move.z + loc_in_neurons, grd);

        if (move.w != -1)
        {
            atomicAdd(wgrd + hl_size * move.w + loc_in_neurons, grd);
        }
    }
}