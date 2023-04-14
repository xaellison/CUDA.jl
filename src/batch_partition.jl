function cumsum!(sums)
    shift = 1

    while shift < length(sums)
        to_add = 0
        @inbounds if threadIdx().x - shift > 0
            to_add = sums[threadIdx().x - shift]
        end

        sync_threads()
        @inbounds if threadIdx().x - shift > 0
            sums[threadIdx().x] += to_add
        end

        sync_threads()
        shift *= 2
    end
end

"""
Partition the region of `values` after index `lo` up to (inclusive) `hi` with
respect to `pivot`. Computes each value's comparison to pivot, performs a cumsum
of those comparisons, and performs one movement using shmem. Comparison is
affected by `parity`. See `flex_lt`. `swap` is an array for exchanging values
and `sums` is an array of Ints used during the merge sort.
Uses block y index to decide which values to operate on.
"""
@inline function batch_partition(values, pivot, swap, sums, lo, hi)
    sync_threads()
    #blockIdx_yz =(blockDim().x * (blockIdx().x - 1)) + threadIdx().x
    idx0 = (blockDim().x * (blockIdx().x - 1)) + threadIdx().x#lo + (blockIdx_yz - 1) * blockDim().x + threadIdx().x
    @inbounds if idx0 <= hi
        val = values[idx0]
        comparison = pivot < val#, parity, lt, by)
    end

    @inbounds if idx0 <= hi
         sums[threadIdx().x] = 1 & comparison
    else
         sums[threadIdx().x] = 1
    end
    sync_threads()

    cumsum!(sums)

    @inbounds if idx0 <= hi
        dest_idx = @inbounds comparison ? blockDim().x - sums[end] + sums[threadIdx().x] : threadIdx().x - sums[threadIdx().x]
        if dest_idx <= length(swap)
            swap[dest_idx] = val
        end
    end
    sync_threads()

    @inbounds if idx0 <= hi
         values[idx0] = swap[threadIdx().x]
    end
    #sync_threads()
end

function old_part(values::AbstractArray{T}, pivot) where {T}
    swap = CuDynamicSharedArray(T, blockDim().x)
    sums = CuDynamicSharedArray(Int, blockDim().x, sizeof(T) * blockDim().x)
    batch_partition(values, pivot, swap, sums, 0, length(values))
    return
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@inline function warp_cumsum(value)
    for i in 0:4
        delta = shfl_up_sync(0xffffffff, value, 1 << i)
        if laneid() > 1 << i
            value += delta
        end
    end
    return value
end


function smartpart(c::AbstractArray{T}, pivot) where T
    atomic_floor = CuStaticSharedArray(Int, 1)
    atomic_ceil = CuStaticSharedArray(Int, 1)
       
    swap = CuDynamicSharedArray(T, blockDim().x)
    # does CuDynamicSharedArray initialize to a value? If so, next block not needed
    @inbounds if threadIdx().x == 1
        atomic_ceil[1] = blockDim().x
        atomic_floor[1] = 0
    end
    sync_threads()
    grid_index = (blockDim().x * (blockIdx().x - 1)) + threadIdx().x
    @inbounds value = c[grid_index]
    comparison = 1
    if grid_index <= length(c)
        comparison = 1 & (value > pivot);        
    end
    lane_cumsum = warp_cumsum(comparison)
    
    sync_warp() # was sync_threads in draft... probably unnesc
    
    # compute warp_floor for all lanes in warp
    warp_floor = -1
    warp_ceil = -1
    warp_sum = -1
    @inbounds if laneid() == 32
        warp_sum = lane_cumsum
        warp_floor = CUDA.@atomic atomic_floor[1] += lane_cumsum
        warp_ceil = CUDA.@atomic atomic_ceil[1] -= warpsize() - lane_cumsum
    end
    sync_warp()
    warp_floor = shfl_sync(0xffffffff, warp_floor, 32)
    warp_ceil = shfl_sync(0xffffffff, warp_ceil, 32)
    warp_sum = shfl_sync(0xffffffff, warp_sum, 32)
    # upload to shmem
    
    false_dest = warp_floor + lane_cumsum
    true_dest = warp_ceil - (laneid() - lane_cumsum) + 1
    dest_idx = Bool(comparison) ? false_dest : true_dest

    if grid_index <= length(c)
        @inbounds swap[dest_idx] = value
    end
    sync_threads()
    # coalesced upload to global
    if grid_index <= length(c)
        @inbounds c[grid_index] = swap[threadIdx().x]
    end
    return
end