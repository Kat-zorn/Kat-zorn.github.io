#define __AVX2_AVAILABLE__

#include <iostream>
#include <cinttypes>
#include <bit>
#include <random>
#include <thread>
#include <chrono>
#include "xoshiro256.hh"

constexpr uint32_t RUNS = 1000000000;
constexpr uint_fast8_t SET_SIZE = 231;
typedef SEFUtility::RNG::Xoshiro256Plus<SIMDInstructionSet::AVX2> AVXrand;

const uint_fast8_t thread_count = std::thread::hardware_concurrency();
// const uint_fast8_t thread_count = 16;

uint_fast8_t roll231(AVXrand &rng);
void roll_thread(uint_fast32_t count, uint_fast8_t *ret);
int main_single();
int main_threaded();

int main()
{
    main_threaded();
}

int main_single()
{
    uint_fast8_t highest;
    roll_thread(RUNS, &highest);
    std::cout << "Highest roll: " << (int)highest << std::endl;
    return 0;
}

int main_threaded()
{
    std::chrono::time_point before = std::chrono::high_resolution_clock::now();

    uint_fast8_t highest;
    uint_fast8_t *ret_array = new uint_fast8_t[thread_count];
    std::thread *threads = new std::thread[thread_count];
    for (uint_fast8_t i = 0; i < thread_count; i++)
    {
        threads[i] = std::thread(roll_thread, RUNS / (uint_fast32_t)thread_count, &ret_array[i]);
    }
    roll_thread(RUNS % thread_count, &highest);
    for (uint_fast8_t i = 0; i < thread_count; i++)
    {
        threads[i].join();
        highest = std::max(highest, ret_array[i]);
    }

    std::chrono::time_point after = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds time_taken_ns = after - before;
    float time_taken_ms = time_taken_ns.count() * 0.000001;
    std::cout << "Highest roll: " << (int)highest << ".\nIt took me: " << time_taken_ms << "ms." << std::endl;
    return 0;
}

void roll_thread(uint_fast32_t count, uint_fast8_t *ret)
{
    uint_fast8_t highest = 0;
    std::random_device rd;
    uint32_t seed[8] = {rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
    std::array<uint64_t, 4> seed_array = *(std::array<uint64_t, 4> *)&seed;
    AVXrand rng(seed_array);
    for (uint_fast32_t i = 0; i < count; i++)
    {
        highest = std::max(highest, roll231(rng));
    }
    *ret = highest;
}

union union_256
{
    __m256i_u d256;
    uint64_t d64[4];
};

uint_fast8_t
roll231(AVXrand &rng)
{
    __m256i_u x = rng.next4();
    __m256i_u y = rng.next4();
    // Hopefully the compiler will precompute this mask.
    __m256i_u mask = {(1 << (256 - 231)) - 1};
    x &= y;
    x &= ~mask;
    // Turn a 256-bit number into several 64-bit numbers.
    union_256 data = {x};
    uint_fast8_t out = std::popcount(data.d64[0]);
    // Counts the amount of ones.
    out += std::popcount(data.d64[1]);
    out += std::popcount(data.d64[2]);
    out += std::popcount(data.d64[3]);
    return out;
}
