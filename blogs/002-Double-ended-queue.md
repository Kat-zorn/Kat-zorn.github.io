# Double-ended Queue

> By Luka Bijma, written 10-01-2025, and last updated 16-01-2025. Published to [my blog](https://blazing-blast.github.io/).

## How a `Vector` works

We are probably all familiar with the `Vector`. In most languages, it is a generic class that implements `push(val)` and `pop()`.
You can `push` values to the end of the `Vector`, and `pop` values from the tail. But, under the hood, the `Vector` does a few more things. It allocates memory for these values in such a way that insertion is in amortized constant time.

I will quickly explain how this works, and provide some implementation examples.
Note that I will be omitting the template magic, and simply assume that we have a type named `T` that we want our vector to act on.
On top of that, I will not be marking everything `const`, and I will not deal with error handling. Lastly, I will use C++ in a C-like way. All of this is to ensure that the concepts that I wish to discuss are clear in the code and not obfuscated by C++‘s unreadability (for people who aren‘t wizards).
A naive vector is defined in the following fashion:

```C++
class VectorT
{
private:
    uint64_t capacity;
    uint64_t length;
    T *data;
public:
    void push(T value);
    T pop();
    VectorT(uint64_t initial_capacity);
    void extend(uint64_t new_capacity);
}
```

`pop` is the most straight-forward to implement.

```C++
T VectorT::pop() {
    if (length == 0) // Very unlikely
    {
        // I also don‘t feel like bothering with error handling,
        // so return null it is.
        return T::NULL;
    }
    return data[--length];
}
```

The `push` implementation follows a similar idea, but we need to consider the case where the capacity of the vector is already filled. In that case, we must first make room for our new data. We double the previous capacity to make sure that we don‘t need to reallocate too often. Other growth factors are possible, but we won‘t consider them.

```C++
void VectorT::push(T value) {
    if (length == capacity) // Very unlikely
    {
        extend(capacity * 2);
    }
    data[length++] = value;
}
```

The constructor is also trivial. Because this is C++, you might want to satisfy the rule of ~~3~~, ~~5~~, 7 if you were to actually implement this.

```C++
VectorT VectorT::VectorT(uint64_t initial_capacity)
{
    data = malloc(initial_capacity);
    capacity = initial_capacity;
    length = 0;
}
```

The `extend` method relies on `realloc(ptr, len)` which tries to extend the allocation passed into it to the desired length, and if that is impossible, it allocates a new buffer of the desired size and copies the old data into it.

```C++
    void VectorT::extend(uint64_t new_capacity)
    {
        data = realloc(data, new_capacity);
        capacity = new_capacity;
    }
```

## The need for `Queue`s

All of this seems to work quite nicely, but it has a limitation. You can only `pop` the most recently added value off the end. So, this data structure is not applicable for first come, first served usage. For this, we can use something called a `Double-ended Queue`. This is a datatructure that on top of `push`ing and `pop`ping from the back, can also do so from the front. We will refer to it as `Queue` from now on, as most implementations of a `Queue` that only implement `push_back` and `pop_front` can be easily made double-sided.

## `Queue` candidate: `Shiftable Vector`

One of the possible datastructures for this application is a `Shiftable Vector`. This is also a `Vector`, but it has the extra methods `shift_left` and `shift_right`, and `pop_front` and `push_front`. These methods can be implemented as follows:

```C++
void ShiftVectorT::shift_left(uint64_t offset)
{
    // memcpy is not used, because data is overlapping.
    memmove(data, &data[offset], (length - offset) * sizeof(T));
}

T ShiftVectorT::pop_front()
{
    if(length == 0) // Very unlikely
    {
        return T::NULL;
    }
    T out = data[0];
    shift_left(1);
    length--;
    return out;
    }
```

The implementation of the `shift_right` and `push_front` methods is similar. As you can see, `push`ing and `pop`ping from the front of a `Shift Vector` incurs a heavy performance penalty, as it includes a `memmove` of nearly the entire vector.

## `Queue` candidate: Using two pointers

To mitigate the need to move memory, we can make it such that there is capacity for the buffer to grow on both sides. We can define our new `Queue` as follows.

```C++
class QueueT
{
private:
    uint64_t capacity;
    uint64_t front_index;
    uint64_t back_index;
    T *data;
public:
    void push_back(T value);
    void push_front(T value);
    T pop_back();
    T pop_front();
    QueueT(uint64_t initial_capacity);
    void extend(uint64_t new_capacity);
    uint64_t length();
}
```

Both `length` and the constructor are trivial to implement. I will omit those implementations here. I will show an implementation of `push_front`. The other `push` and `pop` implementations are very similar.

```C++
void QueueT::push_front(T value)
{
    if (front_index == capacity) // Very unlikely
    {
        extend(capacity * 2);
    }
    data[front_index++] = value;
}
```

The more intersting method is `extend`. Here we need to make sure that there is room to grow on both the front and back.

```C++
void QueueT::extend(uint64_t new_capacity)
{
    if (new_capacity <= capacity) // Very unlikely
    {
        return;
    }

    T* new_buffer = malloc(new_capacity * sizeof(T));
    // Make sure there is equal room on both sides
    uint64_t new_back_index = (new_capacity - length())/2;
    uint64_t new_front_index = new_back_index + length();
    memcpy(&new_buffer[new_back_index], &buffer[back_index], length() * sizeof(T));

    free(data);
    back_index = new_back_index;
    front_index = new_front_index;
    data = new_buffer;
    capacity = new_capacity;
}
```

This solution is already a lot better. It is amortized constant in time (unlike the `Shift Vector`), and it does not do unneeded memory operations.
It does have two slight problems. When the back end runs out of capacity, we should be able to try to only extend the back using `realloc` instead. This sadly doesn‘t work with `realloc`, as the way to copies over the data when the original buffer could not be extended costs us extra time (and forces a `memmove` instead of a `memcpy`). We could write our own version of `realloc` to deal with this, but this `Queue` has worse issues that we should deal with. The second issue is that when you don‘t `push` to the frond and back evenly. This will result in basically wasting half of the `Queue`‘s capacity.

## `Queue` candidate: Ring buffers

To deal with this final problem, we can introduce ring buffers. Instead of treating the buffer like a line segment, we can traverse it like a ring instead. This will bring in some extra (constant time) overhead, but it will prevent the wastage of memory that occurs in our previous attempt. One problem with this is that when the front and back indices are the same, we do not know whether the length is 0 or equal to the capacity. Therefore, we must track this with an extra variable. We will simply add a `bool empty` to the member variables. We also need to specify that `index_front` is the index that will written to next, while `index_back` is the one that will we read from next. This avoids some off-by-one errors. This means that as long as `!empty`, `data[back_index]` will be initialized. Now, the `length` and `push_front` implementations look as follows:

```C++
uint64_t RingQueueT::length()
{
    if (empty) // Very unlikely
    {
        return 0;
    }
    if (front_index == back_index) // Very unlikely
    {
        return capacity;
    }
    // This means that no looping has occurred yet.
    if (front_index > back_index) // Unlikely, because looping starts as soon as data is added to the front.
    {
        return front_index - back_index;
    }
    // The space between the two indices is now the amount of free capacity, not the length.
    return capacity - (front_index - back_index);
}

void RingQueueT::push_front(T value)
{
    // Check whether there is room.
    if(front_index == back_index && !empty) // Very unlikely
    {
        extend(capacity * 2);
    }
    // Write data.
    data[front_index] = value;
    // Increment (and wrap) index.
    if(++front_index == capacity) // Very unlikely
    {
        front_index = 0;
    }
}
```

Although the `length` function is now a bit more convoluted, it won‘t actually be called while `push`ing and `pop`ping, so it doesn‘t matter all too much. On top of that, everything is still in (amortized) constant time.
The `extend` method, on the other hand, does get called while `push`ing, and it gets the same complications as the `length` function. Its new implementation is as follows:

```C++
void RingQueueT::extend(uint64_t new_capacity)
{
    if (new_capacity <= capacity) // Very unlikely
    {
        return;
    }

    T* new_buffer = malloc(new_capacity * sizeof(T));
    uint64_t new_back_index = 0;
    uint64_t new_front_index = length();

    // In this case, we could also just have realloced, but for simplicity, I‘ll keep every branch as similar as possible.
    if (empty); // Very unlikely, only happens in the constructor. No copying needs to be done in this case.
    else if (front_index > back_index) // Very unlikely, will not occur due to other methods in this class, only when a user explicitly calls this method.
    {
        memcpy(&new_buffer[new_back_index], &buffer[back_index], length() * sizeof(T));
    }
    else // Very likely
    {
        // First copy the data from back_index until capacity, then from 0 to front_index
        memcpy(&new_buffer[new_back_index], &buffer[back_index], (capacity - back_index) * sizeof(T));
        memcpy(&new_buffer[capacity - back_index - 1], buffer, front_index * sizeof(T));
    }

    free(data);
    back_index = new_back_index;
    front_index = new_front_index;
    data = new_buffer;
    capacity = new_capacity;
    return;
}
```

As you can see, we no longer have a use for starting the data in the center. It is indeed true that `extend` is less expensive when no looping has occurred, but `extend` will not naturally be called unless the capacity is satiated, which necessarily is when looping has already occurred.

In conclusion, the ring-buffer approach allows us to push and pop from both the front and back, while still making optimal use of the allocated capacity. It does so with nearly no performance penalty compared to traditional `Vector`s (as it still just uses `data[index++]`), but with more expensive `length` calls and direct indexing (as you need to take the index modulo capacity).

Thank you for reading my first proper article. I don‘t enjoy webdev enough to make a comment section, so if you want to say anything, you can do so at [the Github](https://github.com/Blazing-Blast/Blazing-Blast.github.io/), [my email](mailto:spamtheblaze@gmail.com), or Discord at @blazingblast.
