"""
Quicksort!
Alex Ellison
@xaellison

Usage:
quicksort!(my_cuarray)

The main quicksort kernel uses dynamic parallelism. Let's call blocksize M. The
first part of the kernel bubble sorts M elements with maximal stride between
lo and hi. If the sublist is <= M elements, stride = 1 and no recursion
happens. Otherwise, we pick element lo + M / 2 * stride as a pivot. This is
an efficient choice for random lists and pre-sorted lists.

Partition is done in stages:
1. In a separate kernel: Merge-sort batches of M values using their comparison
    to pivot as a key. The comparison alternates between < and <= with recursion
    depth. This makes no difference when there are many unique values, but when
    there are many duplicates, this effectively partitions into <, =, and >.
2. Consolidate batches. This runs inside the quicksort kernel.

Information about the partitions of each batch is stored in shared memory if
there are fewer than N batches (list length <= M ^ 2). Otherwise, it is stored in
a global array. Any sub-kernels which also need that array will be executed
synchronously so they don't conflict over that memory.

Naming conventions:
Sublists (ranges of the list being sorted) are denoted by `my_floor` and one of
    `L` and `my_ceil`. `my_floor` is an exclusive lower bound, `my_ceil` is an
    inclusive upperboard, `L` is their difference.

`lo` / `hi` are used in place of `my_floor` / `my_ceil` in `qsort_kernel`
    `lo` / `hi` are indices of  batches in `consolidate_batch_partition`

`b_sums` is "batch sums", the number of values in a batch which are >= pivot
"""

#-------------------------------------------------------------------------------
# Integer arithmetic
"""
Equivalent to Int(ceil(a / b))
"""
function ceil_div(a, b)
    return a ÷ b + 1 & (a % b != 0)
end

"""
Returns smallest power of 2 >= x
"""
function pow2_ceil(x)
    out = 1
    while out < x
        out *= 2
    end
    out
end

function pow2_floor(x)
    out = 1
    while out * 2 <= x
        out *= 2
    end
    out
end

"""
For a batch of size `n` what is the lowest index of the batch `i` is in
"""
function batch_floor(idx, n)
    return idx - (idx - 1) % n
end

"""
For a batch of size `n` what is the highest index of the batch `i` is in
"""
function batch_ceil(idx, n)
    return idx + n - 1 - (idx - 1) % n
end

"""
GPU friendly step function (step at i = 1)
"""
function Θ(i)
    return 1 & (1 <= i)
end

"""
Suppose we are merging two lists of size n, each of which has all falses before
all trues. Together, they will be indexed 1:2n. This is a fast stepwise function
for the destination index of a value at index `x` in the concatenated input,
where `a` is the number of falses in the first half, b = n - a, and false is the
number of falses in the second half.
"""
function step_swap(x, a, b, c)
    return x + Θ(x - a) * b - Θ(x - (a + c)) * (b + c) + Θ(x - (a + b + c)) * c
end

"""
Generalizes `step_swap` for when the floor index is not 1
"""
function batch_step_swap(x, n, a, b, c)
    idx = (x - 1) % n + 1
    return batch_floor(x, n) - 1 + step_swap(idx, a, b, c)
end

@inline function flex_lt(a, b, eq :: Bool)
    if eq
        return a >= b
    else
        return a > b
    end
end

#-------------------------------------------------------------------------------
# Batch partitioning
"""
For thread `idx` with current value `value`, merge two batches of size `n` and
return the new value this thread takes. `sums` and `swap` are shared mem
"""
function merge_swap_shmem(value, idx, n, sums, swap)
    @inbounds begin
    sync_threads()
    b = sums[batch_floor(idx, 2 * n)]
    a = n - b
    d = sums[batch_ceil(idx, 2 * n)]
    c = n - d
    swap[idx] = value
    sync_threads()
    sums[idx] = d + b
    return swap[batch_step_swap(idx, 2 * n, a, b, c)]
    end
end

@inline function batch_partition(values, swap, pivot, sums, i, idx0, my_floor, L, N_t, parity)
    if idx0 - my_floor <= L
        @inbounds swap[i] = values[idx0]
        @inbounds sums[i] = 1 & flex_lt(swap[i], pivot, parity)
    else
        @inbounds sums[i] = 1
    end
    sync_threads()
    val = merge_swap_shmem(swap[i], i, 1, sums, swap)
    temp = 2
    while temp < N_t
        val = merge_swap_shmem(val, i, temp, sums, swap)
        temp *= 2
    end
    sync_threads()

    if idx0 - my_floor <= L
        @inbounds values[idx0] = val
    end
end

"""
Partition `values` such that all < `pivot` come first, then all >= `pivot`.
NOTE: It is very important that this is a stable sort. We effectively allow
    unallocated memory to be our null values, and set them to true, at the top
    of the list. That way, they can be operated on, but at the end they aren't
    transferred out to global mem.
"""
function partition_batches_kernel(values :: CuDeviceArray{T}, pivot, my_floor, L, parity :: Val{P}) where {T, P}
    N_t = blockDim().x
    i = threadIdx().x
    idx0 = my_floor + (blockIdx().x - 1) * blockDim().x + i # for addressing
    swap = @cuDynamicSharedMem(T, N_t, 4 * N_t)
    sums = @cuDynamicSharedMem(Int32, N_t)
    batch_partition(values, swap, pivot, sums, i, idx0, my_floor, L, N_t, P)
    return nothing
end

"""
Partition `values` such that all < `pivot` come first, then all >= `pivot`.
Method stores the number of values >= `pivot` in each batch in `batch_sums`
"""
function partition_batches_kernel(values :: CuDeviceArray{T}, pivot, batch_sums, my_floor, L, parity :: Val{P}) where {T, P}
    N_t = blockDim().x
    i = threadIdx().x
    idx0 = my_floor + (blockIdx().x - 1) * blockDim().x + i # for addressing
    swap = @cuDynamicSharedMem(T, N_t, 4 * N_t)
    sums = @cuDynamicSharedMem(Int32, N_t)
    batch_partition(values, swap, pivot, sums, i, idx0, my_floor, L, N_t, P)

    if i == 1
        # this is the amount by which our kernel "overhangs" the real list
        # and how many unnecessary trues are included in the sum
        overhang = (idx0 - my_floor <= (L - L % N_t) || L % N_t == 0 ? 0 : N_t - (L % N_t))
        @inbounds batch_sums[blockIdx().x] = sums[i] - overhang
    end
    return nothing
end


#-------------------------------------------------------------------------------
# Batch consolidation
"""
Finds the index in `array` of the last value less than `pivot`.
Searches after index `my_floor` up to (inclusive) index `my_ceil`
"""
function seek_last_less_than(array, pivot, my_floor, my_ceil, parity) :: Int32
    low = my_floor + 1
    high = my_ceil
    while low <= high
        mid = (low + high) ÷ 2
        if flex_lt(array[mid], pivot, parity)
            high = mid - 1
        else
            low = mid + 1
        end
    end
    return low - 1
end

"""
Performs binary search on batches to find their partitions. Stores result in
`b_sums`
"""
@inline function init_bsums(vals, pivot, my_floor, L, N_t, N_b, i, b_sums, parity)
    if i <= N_b
        seek_lo = my_floor + (i - 1) * N_t
        seek_hi = my_floor + min(Int32(L), Int32(i * N_t))
        b_sums[i] = seek_hi - seek_last_less_than(vals, pivot, seek_lo, seek_hi, parity)
    end
    sync_threads()
end

"""
In linear time, merge `N_b` consecutive, partitioned batches so that the
`L` length region of `vals` after `lo` is partitioned. Must only run on 1 SM.
`i` - thread id
`b_sums` - number of values >= `pivot` in each batch
"""
@inline function consolidate_batch_partition(vals, pivot, my_floor, L, N_t, N_b, i, b_sums)
    i = threadIdx().x
    sync_threads()
    b = b_sums[1]
    a = N_t - b
    sync_threads()
    for hi in 2:N_b
        n_eff =  (hi != N_b || L % N_t == 0) ? N_t : L % N_t
        sync_threads()
        d = b_sums[hi]
        c = n_eff - d
        to_move = min(b, c)
        sync_threads()
        if i <= n_eff && i <= to_move
            swap = vals[my_floor + a + i]
        end
        sync_threads()
        if i <= n_eff && i <= to_move
            vals[my_floor + a + i] = vals[my_floor + a + b + c - to_move + i]
        end
        sync_threads()
        if i <= n_eff && i <= to_move
            vals[my_floor + a + b + c - to_move + i] = swap
        end
        sync_threads()
        a += c
        b += d
    end
    sync_threads()
    return my_floor + a
end

#-------------------------------------------------------------------------------
# Sorting
"""
Performs bubble sort on `vals` starting after `lo` and going for min(`L`, 1024)
elements spaced by `stride`. Good for sampling pivot values as well as short
sorts.
"""
@inline function bubble_sort(vals, swap, my_floor, L, i, stride, N_t)
    if i <= L
        L = min(Int32(N_t), Int32(L))
        @inbounds begin
        swap[i] = vals[my_floor + i * stride]
        for level in 0:L
            # get left/right neighbor depending on even/odd level
            buddy = i - 1 + 2 * (1 & (i % 2 != level % 2))
            if 1 <= buddy <= L
                buddy_val = swap[buddy]
            end
            sync_threads()
            if 1 <= buddy <= L
                if (i < buddy) != (swap[i] < buddy_val)
                    swap[i] = buddy_val
                end
            end
            sync_threads()
        end
        vals[my_floor + i * stride] = swap[i]
        end
    end
end

"""
Perform quicksort on `vals` for the region with `lo` as an exclusive floor and
`hi` as an inclusive ceiling.
`block_dim` must be a Val{N_t} where `N_t` must equal the number of threads this
    kernel is launched with.

Assumes that RUNTIME_SYNC_DEPTH = 24. Recursion ends when `depth` = 24, or when
`hi` - `lo` <= `N_t`. If `depth` = 24 but `hi` - `lo` > `N_t`, sort may not be
complete, but further recursion might involve invalid memory accesses.
"""
function qsort_kernel(vals :: CuDeviceArray{T}, lo, hi, global_b_sums, parity :: Val{P}, depth :: Int8) where {T, P}
    #N_t = blockDim().x
    b_sums = @cuDynamicSharedMem(Int32, blockDim().x)
    swap = @cuDynamicSharedMem(T, blockDim().x, 4 * blockDim().x)
    i = threadIdx().x
    L = Int32(hi - lo)
    N_b = ceil_div(L, blockDim().x)

    #= step 1 bubble sort. It'll either finish sorting a subproblem or help
    select a pivot value =#
    bubble_sort(vals, swap, lo, L, i, L <= blockDim().x ? Int32(1) : Int32(L ÷ blockDim().x), blockDim().x)
    if L <= blockDim().x || depth >= 24
        return
    end

    @inbounds pivot = vals[lo + (blockDim().x ÷ 2) * (L ÷ blockDim().x)]

    # step 2: use pivot to partition into batches
    if i == 1
        if L <= blockDim().x ^ 2
            # perform batch partition and do not store partition indices
            @cuda blocks=N_b threads=blockDim().x dynamic=true shmem=blockDim().x*(4+sizeof(T)) partition_batches_kernel(vals, pivot, lo, L, parity)
        else
            # store batch partition indices in global array
            @cuda blocks=N_b threads=blockDim().x dynamic=true shmem=blockDim().x*(4+sizeof(T)) partition_batches_kernel(vals, pivot, global_b_sums, lo, L, parity)
        end
    end
    CUDA.device_synchronize()
    sync_threads()
    #= step 3: consolidate the partitioned batches so that the sublist from
    [lo, hi) is partitioned, and the partition is stored in `partition`=#

    partition = -1
    if L <= blockDim().x ^ 2
        #= if small enough, perform binary search to find the partitions in each
        batch, store in shmem, and use that=#
        init_bsums(vals, pivot, lo, L, blockDim().x, N_b, i, b_sums, P)
        partition = consolidate_batch_partition(vals, pivot, lo, L, blockDim().x, N_b, i, b_sums)
    else
        #= if more batches than threads per SM, use the global array written to
        by `partition_batches_kernel`=#
        partition = consolidate_batch_partition(vals, pivot, lo, L, blockDim().x, N_b, i, global_b_sums)
    end

    #= step 4: recursion. If the first subproblem doesn't require global mem,
    run it on a temporary stream. Otherwise run in default stream, so that
    the two subproblems run serially, and don't conflict over the global array=#
    if i == 1 && partition > lo
        if partition - lo <= blockDim().x ^ 2
            s = CuDeviceStream()
            @cuda threads=blockDim().x dynamic=true stream=s shmem=blockDim().x*(4+sizeof(T)) qsort_kernel(vals, lo, Int32(partition), global_b_sums, Val(!P), depth + Int8(1))
            CUDA.unsafe_destroy!(s)
        else
            @cuda threads=blockDim().x dynamic=true shmem=blockDim().x*(4+sizeof(T)) qsort_kernel(vals, lo, Int32(partition), global_b_sums, Val(!P), depth + Int8(1))
        end
    end

    if i == 1 && partition < hi
        if hi - partition <= blockDim().x ^ 2
            s = CuDeviceStream()
            @cuda threads=blockDim().x dynamic=true stream=s shmem=blockDim().x*(4+sizeof(T)) qsort_kernel(vals, Int32(partition), hi, global_b_sums, Val(!P), depth + Int8(1))
            CUDA.unsafe_destroy!(s)
        else
            @cuda threads=blockDim().x dynamic=true shmem=blockDim().x*(4+sizeof(T)) qsort_kernel(vals, Int32(partition), hi, global_b_sums, Val(!P), depth + Int8(1))
        end
    end
    return nothing
end

function quicksort!(c :: CuArray{T}) where T
    CUDA.limit!(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH, 24)
    N = length(c)

    function get_config(kernel)
        get_shmem(threads) = threads * (sizeof(Int32) + max(4, sizeof(T)))
        fun = kernel.fun
        config = launch_configuration(fun, shmem=threads->get_shmem(threads))
        threads = pow2_floor(config.threads)
        @assert threads <= config.threads
        @assert threads >= 32
        return (blocks=1, threads=threads, shmem=get_shmem(threads))
    end

    b_sums = CuArray{Int32}(undef, ceil_div(N, 32))
    CUDA.@sync(@cuda config=get_config qsort_kernel(c, Int32(0), N, b_sums, Val(true), Int8(1)))
end
 
