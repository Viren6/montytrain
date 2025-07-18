extern "C" __global__ void kernel(
    const int batch_size,
    const int hl_size,
    const int* moves,
    const float* owgt,
    const float* ogrd,
    const float* hl_out,
    float* owgrd,
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

        const float tgrd = ogrd[locmb];
        const float thl = hl_out[plc];

        atomicAdd(owgrd + loc_in_neurons, thl * tgrd);

        const float back_grd = tgrd * owgt[loc_in_neurons];
        const float grd = thl > 0.0F && thl < 1.0F ? 2.0F * sqrtf(thl) * back_grd : 0.0F;

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