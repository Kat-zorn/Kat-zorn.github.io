# Doing stupid things stupidly fast

> Written by Luka Bijma.
> Published to [my blog](https://blazing-blast.github.io/).
> Written on the 19th of January 2025.
> Last updates on the 20th.

## Humble beginnings

On the 10th of August 2024, Austin (ex Game Theory) released a video to his channel ShoddyCast where he wrote some code that runs the following algorithm.

```Text
Roll 1 billion sets of 231 four-sided dice,
and output the most amount of ones rolled in a single set.
```

His code (see below) took 8 days to run this algorithm. He then challenged his viewers to implement a faster version. I, of course, dedicated the next month of my life to it.

```Python
# Sourced from arhourigan/gravel on GitHub.
# Slight adaptations for readability.
import random
from itertools import repeat

items = [1,2,3,4]
numbers = [0,0,0,0]
rolls = 0
maxOnes = 0

while maxOnes < 177 and rolls < 1_000_000_000:
    numbers = [0,0,0,0]
    for i in repeat(None, 231):
        roll = random.choice(items)
        numbers[roll - 1] += 1
    rolls += 1
    maxOnes = max(maxOnes, numbers[0])
print(f"Highest roll: {maxOnes}.\nNumber of rolls: {rolls}.")
```

The main thing that jumps out at me when reading this code is the random.choice(items). But as the code is filled with many
inefficiencies in general, I will simply rewrite it in C++ and add comments for each change I made.

```C++
#include <iostream>
#include <cinttypes>
#include <random>

constexpr uint32_t RUNS = 1_000_000_000;
constexpr uint_fast8_t SET_SIZE = 1_000_000_000;


int main()
{
    uint_fast8_t highest = 0;
    uint_fast8_t current = 0;
    // A true RNG used to seed the pseudo-RNG.
    std::random_device rd;
    std::mt19937_64 rng(rd());
    for (i = 0; i < RUNS; i++)
    {
        current = 0;
        for (j = 0; j < SET_SIZE; j++)
        {
            // Checks whether the last 2 bits are 0.
            // This is a 1/4 chance.
            // As true == 1 in C(++), we can just add it.
            current += (rng() & 3) == 0;
        }
        // Use built-in max to aid the compiler.
        highest = std::max(highest, current);
    }
    // The (int) cast is needed because otherwise it will print as a char.
    std::count << "Highest roll: " << (int)highest << std::endl;
    return 0;
}
```

We will be compiling this and all following C++ programs with `-Ofast -march=native -g -std=C++20`.
This version already does 10 million runs in 3.57 seconds. Extrapolating gives 6 minutes for the full 1 billion.
That is already a massive improvement, but it is far from the lowest we can go. When using callgrind, we observe that around 6% of the CPU time is spent just doing the looping (which we will look into later), but nearly all the rest of it is spent inside the random number generator. This gives us two obvious paths forwards: generate random numbers faster, and generate less random numbers.

## Generating less random numbers

We currently generate a 64-bit random number, but we only use the last 2 bits. This seems inefficient. To make use of this new way, we can repeatedly shift the bits, and consider the last 2. This looks as follows

```C++
int roll32(std::mt19937_64 &rng)
{
    uint_fast8_t count;
    uint64_t x = rng();
    for (uint_fast8_t i = 0; i < 32; i++)
    {
        count += !(x & 3);
        x >>= 2;
    }
    return count;
}
```

Through testing, I have found that using a 32-bit random number and manually unrolling this loop is most efficient, but we will forgo optimizing to this level until we have already optimized all the ‘easy’ inefficiencies. Using this, the code comes out to the following.

```C++
int main()
{
    uint_fast8_t highest = 0;
    uint_fast8_t current = 0;
    uint64_t x;
    std::random_device rd;
    std::mt19937_64 rng(rd());
    for (uint_fast32_t i = 0; i < RUNS; i++)
    {
        current = 0;
        for (uint_fast8_t j = 0; j < SET_SIZE / 32; j++)
        {
            current += roll32(rng);
        }
        x = rng();
        // We do not have a multiple of 32 as our SET_SIZE.
        // We should also take care of the remainder.
        for (uint_fast8_t i = 0; i < SET_SIZE % 32; i++)
        {
            current += !(x & 3);
            x >>= 2;
        }
        highest = std::max(highest, current);
    }
    std::cout << "Highest roll: " << (int)highest << std::endl;
    return 0;
}
```

Now the timeshare of the looping is close to 40%, out of a total 85s seconds. If only there was a way to check whether two bits were set by entire registers at once... Oh wait, that is just a logical AND. Indeed, all we need to do is

```C++
int roll64(std::mt19937_64 &rng)
{
    uint64_t x = rng();
    uint64_t y = rng();
    // Counts the amount of ones.
    return std::popcount(x & y);
}
```

This is such an elegant solution, but I did not come up with it. The credit for that goes to Discord user `blue2113` on Austin's Discord server. This way, we can work so much faster. The `main` function now looks as follows

```C++
int main()
{
    uint_fast8_t highest = 0;
    uint_fast8_t current = 0;
    uint64_t x;
    uint64_t y;
    std::random_device rd;
    std::mt19937_64 rng(rd());
    for (uint_fast32_t i = 0; i < RUNS; i++)
    {
        current = 0;
        for (uint_fast8_t j = 0; j < SET_SIZE / 64; j++)
        {
            current += roll64(rng);
        }
        // Set all but the last 2 * (SET_SIZE % 64) bits to 0.
        // The order of operation when it comes to binary operators
        // is a bit strange in C++, so I'll just add parentheses everywhere.
        x = rng() & rng() & (2 * (1ULL << (SET_SIZE % 64)) - 1);
        current += std::popcount(x);
        highest = std::max(highest, current);
    }
    std::cout << "Highest roll: " << (int)highest << std::endl;
    return 0;
}
```

This way, we can now do the full 1 billion runs in just 11.5 seconds. Already, this seems amazing, but we can go much further. Let's look at what Callgrind has to say now. It shows that we are spending 80% of our time in the RNG. This means that we should probably look into which RNG to use.

## Choosing a random number generator

See here, the fastest random number generator:

```C++
int random()
{
    // Randomly chosen by a dice roll.
    return 4;
}
```

Using this generator would not constitute fair play. Neither is using this one

```C++
int seed;
int random()
{
    seed ^= -1;
    return seed;
}
```

Even though every bit will have a 50-50 split, the bits are not independent of each-other.
To not have to deal with the question “What is random enough?” we (the community trying to solve this) can either agree to all use the same random number generator, or agree to have it always return 0, and use something like Rust's `std::black_box` to stop optimization of those calls. The decision was made to use `Wyrand` and `Xoshiro`. This means that we simply replace `std::mt177937_64` by `wy::rand`. So we replace

```C++
std::random_device rd;
std::mt19937_64 rng(rd());
```

by

```C++
wy::rand rng;
```

This cuts our time in half, to 4 seconds for 1 billion iterations.
We currently spend 6% of our time just doing the for loop, 9% on `popcount`ing, 4% on the `and`, and another 4% on the max. The other 75% of the time is spent generating random numbers. As we already make full use of the entropy that the generator gives us, and we can only use one of two generators, we must either look into other methods, or try the `Xoshiro`.

## Going on a Xoshiro sidequest

`Xoshiro` has one great power: it comes in 128, 256, and even 512-bit variants. Let's pick the biggest one, and start writing.
Sadly, this does not work. My CPU is not capable of AVX-512, and neither were the other machines that I tried, so we will have to settle with `Xoshiro256+`.
First, we select the SIMD instruction set that we want to use. We do so using `typedef SEFUtility::RNG::Xoshiro256Plus<SIMDInstructionSet::AVX2> AVXrand`. This way, we can use `AVXrand` to refer to the generator's type.
We then initialize a generator using:

```C++
std::random_device rd;
const uint32_t seed[2] = {rd(), rd()};
AVXrand rng((uint64_t)seed);
```

We roll all the 231 dice at once using

```C++
__m256i_u x = rng.next4();
__m256i_u y = rng.next4();
// Hopefully the compiler will precompute this mask.
__m256i_u mask = {(1 << (256 - 231)) - 1};
x &= y;
x &= ~mask;
```

Then, we need to `popcount` them, but the C++ standard library does not properly implement `popcount` for values above 64-bit. Instead, we could use the one provided by AVX512. That would look like `x = _mm512_popcnt_epi32(x)`, which counts the bits per `int`, and then use `_mm512_reduce_add_epi32(x)`. Sadly, I don't have access to AVX512, so I will need to split the vector register into its components and popcount and sum them separately. Here's how

```C++
// Turn a 256-bit number into several 64-bit numbers.
union_256 data = {x};
uint_fast8_t out = std::popcount(data.d64[0]);
// Counts the amount of ones.
out += std::popcount(data.d64[1]);
out += std::popcount(data.d64[2]);
out += std::popcount(data.d64[3]);
return out;
```

Where `union_256` is defined as

```C++
union union_256 {
    __m256i_u d256;
    uint64_t d64[4];
};
```

This results in a time of 3.7s, which is less, but not as much less as I had hoped. Let's look at what Callgrind says. It gives 23% of time spend in `popcount`, 5% spend looping, and 55% of time generating random numbers. This does not seem right, as we are `popcount`ing just as often as we did before, so I think that we are surpassing my level of expertise using Callgrind. Now that we are generating random numbers as fast as possible, and extracting the data as fast as possible, there is only one channel of optimization left: generating multiple random numbers at once.

## Parallelizing random number generation

By virtue of all our rolls being completely independent of each-other, this problem is really well-suited for parallelization. Because I have no use for the overhead that promises bring, I will define the function that runs on each thread as follows:

```C++
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
```

Then, the `main` function will create an array for return values, launch as many threads as the system supports, and compute the maximum value of the return array after all threads are finished. To find the amount of threads, we globally define `const uint_fast8_t thread_count = std::thread::hardware_concurrency();`. Then the new `main` function will look like this:

```C++
main_threaded()
{
    uint_fast8_t highest;
    uint_fast8_t *ret_array = new uint_fast8_t[thread_count];
    std::thread *threads = new std::thread[thread_count];
    // These threads automatically start once initialized
    for (uint_fast8_t i = 0; i < thread_count; i++)
    {
        threads[i] = std::thread(roll_thread, RUNS / (uint_fast32_t)thread_count, &ret_array[i]);
    }
    // Finish up the remaining rolls
    roll_thread(RUNS % thread_count, &highest);
    // Compute the max
    for (uint_fast8_t i = 0; i < thread_count; i++)
    {
        threads[i].join();
        highest = std::max(highest, ret_array[i]);
    }
    std::cout << "Highest roll: " << (int)highest << std::endl;
    return 0;
}
```

This is a big improvement again. I get around 0.38 seconds now. But at this timescale, the problems of the `time` command start to show. It is only accurate to a few dozen milliseconds, which is no longer enough. Instead, we will time ourselves using `Chrono`, C++'s time library, which allows us to record time with nanosecond precision (if your hardware clock can as well). To start timing, we use `std::chrono::time_point before = std::chrono::high_resolution_clock::now()`, and we also record an `after` time. Then, we get the time using `(after - before).count()` and multiply by $$10^{-6}$$ to get the number in milliseconds. After printing this out, we can see the time a lot more accurately. For this program, we now take from 418ms to 359ms to run the program. There is still a lot of variance, but I will go by “best run counts.” Therefore, this version performs at 359ms. If we go even faster, we might need to consider more than 1 billion rolls to be able to measure accurately.

## Going even faster

Seeing as a few dozen threads weren't an issue, how about a few thousand? Indeed, it is time to rewrite all of our code to run on the GPU. Although I originally did my transition to the GPU in CUDA, my Radeon is more powerful than any of my NVIDIA GPUs, so I will rewrite it in HIP. I have never worked with HIP, but it seems that it is comes with a CUDA-to-HIP transpiler. I do want to mention another reason for switching away from CUDA, that is that proper profiling tools are only available for Windows, so I do not have access to them on Arch (btw).

First, we should find a way to generate random numbers in HIP. We could use built-in methods, or port Wyrand to it. As this is a function that will run on the GPU, we need to specify that with `__device__`.

```C++
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
```

This function is stitched together from the many one-line functions inside the `Wyrand` source code. Then, we will create our `roll_hip` function. As we don't want to compute a true random seed for every thread on the GPU
