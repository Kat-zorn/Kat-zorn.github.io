#include <iostream>
#include <cinttypes>
#include <random>
#include <chrono>

constexpr uint_fast8_t totalTurns = 231;
constexpr uint64_t maxRuns = 100000000ULL;
constexpr int blockSize = 1 << 10;
constexpr int blockCount = 1 << 10;
constexpr int totalThreads = blockSize * blockCount;

__always_inline int main_unthreaded();
__always_inline int main_threaded();
__global__ void roll_cuda(uint_fast8_t *ret, uint64_t seed);
__device__ static __always_inline uint64_t wyrand(uint64_t &seed);

int main()
{
    std::random_device rd;
    uint_fast8_t highest = 0;
    uint_fast8_t *results;
    cudaEvent_t start, stop;
    float time;
    const uint64_t seed = ((uint64_t)rd() << 32) | rd();

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaMallocManaged(&results, blockCount * blockSize * sizeof(*results));

    roll_cuda<<<blockCount, blockSize>>>(results, seed);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    for (size_t i = 0; i < blockCount * blockSize; i++)
    {
        highest = std::max(highest, results[i]);
    }

    std::cout << "My record is: " << (int)highest << ".\nIt took me " << time << "ms.\n";
    return 0;
}

__global__ void roll_cuda(uint_fast8_t *results, uint64_t seed)
{
    uint_fast8_t highest = 0;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    uint_fast8_t current;
    seed += index;

    const size_t runs = (maxRuns / totalThreads) + (index < maxRuns % (totalThreads));
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

__device__ static __always_inline uint64_t wyrand(uint64_t &seed)
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
