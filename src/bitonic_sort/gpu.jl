using CUDA

"""
This is an iterative bitonic sort which mimics a recursive version to support
non-power2 lengths.

Credit for the recursive form of this algorithm goes to:
https://www.inf.hs-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm

CUDA.jl implementation originally by @xaellison

Overview: The recursive bitonic sort is like most recursive sorts: it does
something to a range of values [a_1, a_2, ... a_n] and then operates on
two subregions [a_1, ... a_n÷2] and [a_n÷2, ... a_n]. To port this to GPU, each
thread of the grid has to decide which recursive region it belongs in. The
simpler and more general way is to have each grid thread i responsible for
swapping c[i] in CuArray c. `kernel` does this but that  requires a kernel
launch per level of swaps to avoid race conditions.

However, many of these swaps can be bundled into a single kernel if they don't
need to swap outside the range handled by a block. This is what `kernel_small`
does. Rather than use each grid index to determine the range, the index of the
block determines the range that all threads in the block share. Within that
range consecutive threads operate on consecutive array indices.
"""

# General functions

@inline function gp2lt(x :: Int)
   x -= 1
   x |= x >> 1
   x |= x >> 2
   x |= x >> 4
   x |= x >> 8
   x |= x >> 16
   x |= x >> 32
   xor(x, x >> 1)
end

@inline function exchange(A :: AbstractArray{T}, i, j) where T
    @inbounds A[i + 1], A[j + 1] = A[j + 1], A[i + 1]
end

@inline function compare(A :: AbstractArray{T}, i, j, dir :: Bool, by, lt) where T
    @inbounds if dir != lt(by(A[i + 1]) , by(A[j + 1]))
        exchange(A, i, j)
    end
end

@inline function compare(A_I :: Tuple{AbstractArray{T}, AbstractArray{Int}}, i, j, dir :: Bool, by, lt) where {T}
    A, i_A = A_I
    @inbounds if dir != lt(by(A[i + 1]) , by(A[j + 1]))
        exchange(A, i, j)
        exchange(i_A, i, j)
    end
end

@inline function compare_large(A :: AbstractArray{T}, i, j, dir :: Bool, by, lt) where {T}
    @inbounds if dir != lt(by(A[i + 1]) , by(A[j + 1]))
        exchange(A, i, j)
    end
end

@inline function compare_large(A_I :: Tuple{AbstractArray{T}, AbstractArray{Int}}, i, j, dir :: Bool, by, lt) where {T}
    A, i_A = A_I
    @inbounds if dir != lt(by(A[i_A[i + 1]]) , by(A[i_A[j + 1]]))
        exchange(i_A, i, j)
    end
end


# Functions specifically for "large" bitonic steps (those that cannot use shmem)


@inline function get_range_part1(L, index, depth1)
    lo = 0
    dir = true
    for iter in 1:depth1-1
        if L <= 1
            return -1, -1, false
        end

        if index < lo + L ÷ 2
            L = L ÷ 2
            dir = !dir
        else
            lo = lo + L ÷ 2
            L = L - L ÷ 2
        end
    end
    return lo, L, dir
end

@inline function evolver(index, L, lo)
    if L <= 1
        return -1, -1
    end
    m = gp2lt(L)
    if index < lo + m
        L = m
    else
        lo = lo + m
        L = L - m
    end
    return lo, L
end

@inline function get_range_part2(lo, L, index, depth2)
    for iter in 1:depth2-1
        lo, L = evolver(index, L, lo)
    end
    return lo, L
end

"""
Determines parameters for swapping when the grid index directly maps to an
Array index for swapping
"""
@inline function get_range(L, index , depth1, depth2)
    lo, L, dir = get_range_part1(L, index, depth1)
    lo, L = get_range_part2(lo, L, index, depth2)
    return lo, L, dir
end

"""
Performs a step of bitonic sort requiring swaps between indices further apart
than the size of block allows (eg, 1 <--> 10000)

The grid index directly maps to the index of `c` that will be used in the swap.

Note that to avoid synchronization issues, only one thread from each pair of
indices being swapped will actually move data. This does mean half of the threads
do nothing, but it works for non-power2 arrays while allowing direct indexing.
"""
function kernel(c, length_c, depth1, depth2, by::F1, lt::F2) where {F1, F2}
    index = (blockDim().x * (blockIdx().x - 1) ) + threadIdx().x - 1

    lo, n, dir = get_range(length_c, index, depth1, depth2)

    if ! (lo < 0 || n < 0) && !(index >= length_c)

        m = gp2lt(n)
        if  lo <= index < lo + n - m
            i, j = index, index + m
            @inbounds compare_large(c, i, j, dir, by, lt)
        end
    end
    return
end


# Functions for "small" bitonic steps (those that can use shmem)


"""
For each thread in the block, "re-compute" the range which would have

been passed in recursively. This range only depends on the block, and guarantees
all threads perform swaps accessible using shmem.

Various negative exit values just for debugging.
"""
function block_setup(L, block_index, depth1, depth2)
    lo = 0
    dir = true
    tmp = block_index * 2
    for iter in 1:(depth1-1)
        tmp ÷= 2
        if L <= 1
            return -1, -1, false
        end

        if tmp % 2 == 0
            L = L ÷ 2
            dir = !dir
        else
            lo = lo + L ÷ 2
            L = L - L ÷ 2
        end
    end

    for iter in 1:(depth2-1)
        tmp ÷= 2
        if L <= 1
            return -2, -2, false
        end

        m = gp2lt(L)
        if tmp % 2 == 0
            L = m
        else
            lo = lo + m
            L = L - m
        end

    end
    if 0 <= L <= 1
        return -3, -3, false
    end
    return lo, L, dir
end

"""
For sort/sort! `c`, allocate and return shared memory view of `c`
"""
function initialize_shmem(c :: AbstractArray{T}, index, in_range, offset=0) where T
    swap = CuDynamicSharedArray(T, (blockDim().x, blockDim().y), offset)
    if in_range
        @inbounds swap[threadIdx().x, threadIdx().y] = c[index + 1]
    end
    sync_threads()
    return @view swap[:, threadIdx().y]
end

"""
For sortperm/sortperm!, allocate and return shared memory views of `c` and index
array
"""
function initialize_shmem(c :: Tuple{AbstractArray{T}, AbstractArray{Int}}, index, in_range) where T
    swap_vals = CuDynamicSharedArray(T, (blockDim().x, blockDim().y))
    swap_indices = initialize_shmem(c[2], index, in_range, sizeof(swap_vals))
    if in_range
        @inbounds swap_vals[threadIdx().x, threadIdx().y] = c[1][swap_indices[threadIdx().x]]
    end
    sync_threads()
    vals_view = @view swap_vals[:, threadIdx().y]
    return vals_view, swap_indices
end

"""
For sort/sort!, copy shmem array `swap` back into global array `c`
"""
function finalize_shmem(c :: AbstractArray{T}, swap :: AbstractArray{T}, index, in_range) where T
    if in_range
        @inbounds c[index + 1] = swap[threadIdx().x]
    end
end

"""
For sortperm/sortperm!, copy shmem array `swap` back to global index array
"""
function finalize_shmem(c :: Tuple{AbstractArray{T}, AbstractArray{Int}},
                        swap :: Tuple{AbstractArray{T}, AbstractArray{Int}},
                        index,
                        in_range) where T
    finalize_shmem(c[2], swap[2], index, in_range)
end

"""
Performs consecutive steps of bitonic sort requiring swaps between indices no
further apart than the size of block allows. This effectively moves part of the
inner loop (over log_j, below) inside of a kernel to minimize launches and do
swaps in shared mem.
"""
function kernel_small(c :: Union{AbstractArray{T}, Tuple{AbstractArray{T}, AbstractArray{Int}}},
                      length_c, depth1, d2_0, d2_f, by::F1, lt::F2) where {T,F1,F2}
    pseudo_block_idx = (blockIdx().x - 1) * blockDim().y + threadIdx().y - 1
    _lo, _n, dir = block_setup(length_c, pseudo_block_idx, depth1, d2_0)
    index = _lo + threadIdx().x - 1
    in_range = threadIdx().x <= _n && _lo >= 0

    swap = initialize_shmem(c, index, in_range)

    # mutable copies
    lo, n = _lo, _n

    for depth2 in d2_0:d2_f
        if ! (lo < 0 || n < 0) && in_range

            m = gp2lt(n)
            if  lo <= index < lo + n - m
                i, j = index - _lo, index - _lo + m
                compare(swap, i, j, dir, by, lt)
            end
        end
        lo, n = evolver(index, n, lo)
        sync_threads()
    end

    finalize_shmem(c, swap, index, in_range)
    return
end


# Host side code


function bitonic_sort!(c :: AbstractArray{T}; by=identity, lt=isless) where T
    log_k0 = c |> length |> log2 |> ceil |> Int

    # These two outer loops are the same as the serial version outlined here:
    # https://en.wikipedia.org/wiki/Bitonic_sorter#Example_code
    for log_k in log_k0:-1:1

        j_final = (1+log_k0-log_k)

        for log_j in 1:j_final

            get_shmem(threads) = prod(threads) * (sizeof(Int) + sizeof(T))
            args1 = (c, length(c), log_k, log_j, j_final, by, lt)
            kernel1 = @cuda launch=false kernel_small(args1...)
            config1 = launch_configuration(kernel1.fun, shmem=threads->get_shmem(threads))
            threads1 = prevpow(2, config1.threads)

            args2 = (c, length(c), log_k, log_j, by, lt)
            kernel2 = @cuda launch=false kernel(args2...)
            config2 = launch_configuration(kernel2.fun, shmem=threads->get_shmem(threads))
            threads2 = prevpow(2, config2.threads)

            threads = min(threads1, threads2)
            log_threads = threads |> log2 |> Int

            if log_k0 - log_k - log_j + 2 <= log_threads
                pseudo_block_size = 1 << abs(j_final + 1 - log_j)
                _block_size = if pseudo_block_size < 256
                    (pseudo_block_size, 256 ÷ pseudo_block_size)
                else
                    (pseudo_block_size, 1)
                end
                b = nextpow(2, cld(length(c), prod(_block_size)))
                kernel1(args1...; blocks=b, threads=_block_size, shmem=get_shmem(_block_size))
                break
            else
                b = nextpow(2, cld(length(c), threads))
                kernel2(args2...; blocks=b, threads=threads)
            end

        end

    end
end

"""
Basically identical to `bitonic_sort` but passes a tuple of the CuArray of values
with an Array of indices
"""
function bitonic_sortperm!(I, c :: AbstractArray{T}; by=identity, lt=isless) where T

    log_k0 = c |> length |> log2 |> ceil |> Int

    for log_k in log_k0:-1:1

        j_final = (1 + log_k0 - log_k)

        for log_j in 1:j_final

            get_shmem(threads) = prod(threads) * (sizeof(Int) + sizeof(T))
            args1 = ((c, I), length(c), log_k, log_j, j_final, by, lt)
            kernel1 = @cuda launch=false kernel_small(args1...)
            config1 = launch_configuration(kernel1.fun, shmem=threads->get_shmem(threads))
            threads1 = prevpow(2, config1.threads)

            args2 = ((c, I), length(c), log_k, log_j, by, lt)
            kernel2 = @cuda launch=false kernel(args2...)
            config2 = launch_configuration(kernel2.fun, shmem=threads->get_shmem(threads))
            threads2 = prevpow(2, config2.threads)

            threads = min(threads1, threads2)
            log_threads = threads |> log2 |> Int

            if log_k0 - log_k - log_j + 2 <= log_threads
                pseudo_block_size = 1 << abs(j_final + 1 - log_j)
                _block_size = if pseudo_block_size < 256
                    (pseudo_block_size, 256 ÷ pseudo_block_size)
                else
                    (pseudo_block_size, 1)
                end
                b = nextpow(2, cld(length(c), prod(_block_size)))
                kernel1(args1...; blocks=b, threads=_block_size, shmem=get_shmem(_block_size))
                break
            else
                b = nextpow(2, cld(length(c), threads))
                kernel2(args2...; blocks=b, threads=threads)
            end
        end
    end
    return I
end

a = rand(Float32, 100_000)
c = CuArray(a)
I = CuArray(collect(1:length(c)))
bitonic_sortperm!(I, c)
synchronize()
