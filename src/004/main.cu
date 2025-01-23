#include "hip/hip_runtime.h"
#include <iostream>
#include <cinttypes>
#include <random>
#include <chrono>

constexpr uint_fast8_t totalTurns = 231;
constexpr uint64_t maxRuns = 100000000ULL;
constexpr int blockSize = 1 << 10;
constexpr int blockCount = 1 << 10;
constexpr int totalThreads = blockSize * blockCount;

__global__ void roll_hip(uint_fast8_t *ret, uint64_t seed);
__device__ uint64_t wyrand(uint64_t &seed);
void hip_catch(hipError_t err);

int main()
{
    std::random_device rd;
    uint_fast8_t highest = 0;
    uint_fast8_t *results;
    hipEvent_t start, stop;
    float time;
    const uint64_t seed = ((uint64_t)rd() << 32) | rd();

    hip_catch(hipEventCreate(&start));
    hip_catch(hipEventCreate(&stop));
    hip_catch(hipEventRecord(start, 0));
    hip_catch(hipMallocManaged(&results, blockCount * blockSize * sizeof(*results)));

    roll_hip<<<blockCount, blockSize>>>(results, seed);
    hip_catch(hipEventRecord(stop, 0));
    hip_catch(hipEventSynchronize(stop));
    hip_catch(hipEventElapsedTime(&time, start, stop));

    for (size_t i = 0; i < totalThreads; i++)
    {
        highest = std::max(highest, results[i]);
    }

    std::cout << "My record is: " << (int)highest << ".\nIt took me " << time << "ms.\n";
    return 0;
}
void hip_catch(hipError_t err)
{
    if (err)
    {
        std::cerr << "HIP errorcode "
                  << err << '\n';
    }
}

__global__ void roll_hip(uint_fast8_t *results, uint64_t seed)
{
    uint_fast8_t highest;
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
