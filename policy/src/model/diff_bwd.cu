extern "C" __global__ void kernel(
    const int batch_size,
    const int hl_size,
    const int* moves,
    const float* ogrd,
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

    int4 move = reinterpret_cast<const int4*>(moves)[64 * loc_in_batch + loc_in_moves];

    if (move.x + move.y + move.z + move.w != -4) {
        const float grd = ogrd[64 * hl_size * loc_in_batch + hl_size * loc_in_moves + loc_in_neurons];

        atomicAdd(hgrd + hl_size * loc_in_batch + loc_in_neurons, grd);

        if (move.x != -1)
        {
            atomicAdd(wgrd + hl_size * move.x + loc_in_neurons, grd);
        }

        if (move.y != -1)
        {
            atomicAdd(wgrd + hl_size * move.y + loc_in_neurons, grd);
        }

        if (move.z != -1)
        {
            atomicAdd(wgrd + hl_size * move.z + loc_in_neurons, grd);
        }

        if (move.w != -1)
        {
            atomicAdd(wgrd + hl_size * move.w + loc_in_neurons, grd);
        }
    }
}