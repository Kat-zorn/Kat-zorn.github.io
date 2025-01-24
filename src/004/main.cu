#include "hip/hip_runtime.h"
#include <iostream>
#include <cinttypes>
#include <random>
#include <chrono>

constexpr uint_fast8_t totalTurns = 231;
constexpr uint64_t maxRuns = 1000000000ULL;
constexpr uint32_t blockSize = 1 << 10;
constexpr uint32_t blockCount = 1 << 10;
constexpr uint64_t totalThreads = blockSize * blockCount;

__global__ void roll_hip(uint_fast8_t *ret, uint64_t seed);
__global__ void maximum_reduce(uint_fast8_t *src, uint_fast8_t *dest);
__device__ uint64_t wyrand(uint64_t &seed);
void hip_catch(hipError_t err);

int main()
{
    uint_fast8_t highest = 0;
    uint_fast8_t *results;
    uint_fast8_t *results_condensed;

    hipEvent_t start;
    hipEvent_t stop;
    float time_gpu;

    std::random_device rd;
    const uint64_t seed = ((uint64_t)rd() << 32) | rd();

    hip_catch(hipEventCreate(&start));
    hip_catch(hipEventCreate(&stop));
    hip_catch(hipMallocManaged(&results, totalThreads * sizeof(*results)));
    hip_catch(hipMallocManaged(&results_condensed, blockCount * sizeof(*results)));

    hip_catch(hipEventRecord(start, 0));
    roll_hip<<<blockCount, blockSize>>>(results, seed);
    maximum_reduce<<<1, blockCount>>>(results, results_condensed);
    hip_catch(hipEventRecord(stop, 0));

    hip_catch(hipEventSynchronize(stop));
    hip_catch(hipEventElapsedTime(&time_gpu, start, stop));

    std::chrono::time_point before = std::chrono::high_resolution_clock::now();

    // We only need to fin the maximum in the condensed array.
    for (size_t i = 0; i < blockCount; i++)
    {
        highest = std::max(highest, results_condensed[i]);
    }

    std::chrono::time_point after = std::chrono::high_resolution_clock::now();
    float time_cpu = (after - before).count() * 0.000001;

    std::cout << "My record is: " << (int)highest << " .\n"
              << "It took me " << time_gpu << "ms on the GPU.\n"
              << "And " << time_cpu << "ms on the CPU.\n";
    return 0;
}

void hip_catch(hipError_t err)
{
    if (err)
    {
        std::cerr << "HIP errorcode " << err << '\n';
    }
}

__global__ void roll_hip(uint_fast8_t *results, uint64_t seed)
{
    uint_fast8_t highest = 0;
    uint_fast8_t current;
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    seed += index;
    const size_t runs = (maxRuns / totalThreads) + (index < (maxRuns % totalThreads));
    for (size_t i = 0; i < runs; i++)
    {
        current = __popcll(wyrand(seed) & wyrand(seed));
        current += __popcll(wyrand(seed) & wyrand(seed));
        current += __popcll(wyrand(seed) & wyrand(seed));
        current += __popcll(wyrand(seed) & wyrand(seed) << (256 - (totalTurns % 256)));
        highest = max(highest, current);
    }
    results[index] = highest;
}

__global__ void maximum_reduce(uint_fast8_t *src, uint_fast8_t *dest)
{
    const uint32_t index = threadIdx.x;
    uint_fast8_t out = 0;
    for (size_t i = index * blockCount; i < (index + 1) * blockCount; i++)
    {
        out = max(out, src[i]);
    }
    dest[index] = out;
}

__device__ uint64_t wyrand(uint64_t &seed)
{
    seed += 0xa0761d6478bd642full;
    uint64_t A = seed,
             B = seed ^ 0xe7037ed1a0b428dbull;
    __uint128_t r = A;
    r *= B;
    A = (uint64_t)r;
    B = (uint64_t)(r >> 64);
    return A ^ B;
}
