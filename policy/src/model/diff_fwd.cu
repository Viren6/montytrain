__forceinline__ __device__ void addf4(float4* dst, const float4 src)
{
    dst->x += src.x;
    dst->y += src.y;
    dst->z += src.z;
    dst->w += src.w;
}

__forceinline__ __device__ void subf4(float4* dst, const float4 src)
{
    dst->x -= src.x;
    dst->y -= src.y;
    dst->z -= src.z;
    dst->w -= src.w;
}

__forceinline__ __device__ float op(const float x)
{
    const float clamp = max(min(x, 1.0F), 0.0F);
    return clamp * clamp;
}

extern "C" __global__ void kernel(
    const int batch_size,
    const int hl_size,
    const float* weights,
    const float* hl,
    const int* moves,
    float* output
) {
    const int loc_in_batch = blockIdx.z;
    const int loc_in_moves = blockIdx.y;
    const int loc_in_neurons = blockIdx.x * blockDim.x + threadIdx.x;

    if (4 * loc_in_neurons >= hl_size || loc_in_batch >= batch_size || loc_in_moves >= 64)
    {
        return;
    }

    const int locmb = 64 * loc_in_batch + loc_in_moves;
    const int4 move = reinterpret_cast<const int4*>(moves)[locmb];

    float4 val = make_float4(0.0F, 0.0F, 0.0F, 0.0F);

    if (move.x != -1) {
        val = reinterpret_cast<const float4*>(hl + hl_size * loc_in_batch)[loc_in_neurons];
        subf4(&val, reinterpret_cast<const float4*>(weights + hl_size * move.x)[loc_in_neurons]);

        if (move.y != -1)
        {
            subf4(&val, reinterpret_cast<const float4*>(weights + hl_size * move.y)[loc_in_neurons]);
        }

        addf4(&val, reinterpret_cast<const float4*>(weights + hl_size * move.z)[loc_in_neurons]);

        if (move.w != -1)
        {
            addf4(&val, reinterpret_cast<const float4*>(weights + hl_size * move.w)[loc_in_neurons]);
        }

        val.x = op(val.x);
        val.y = op(val.y);
        val.z = op(val.z);
        val.w = op(val.w);
    }

    reinterpret_cast<float4*>(output + hl_size * locmb)[loc_in_neurons] = val;
}