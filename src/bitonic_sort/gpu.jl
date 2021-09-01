using CUDA

function gp2lt(n)
    if n <= 1
        return 0
    end
    out = 1
    while out * 2 < n
        out *= 2
    end
    out
end

@inline exchange(A, i, j) = begin A[i + 1], A[j + 1] = A[j + 1], A[i + 1] end

@inline compare(A, i, j, dir :: Bool) = if dir == (A[i + 1] > A[j + 1]) exchange(A, i, j) end

@inline function get_range_part1(L, index, depth1) :: Tuple{Int, Int, Bool}
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

@inline function get_range_part2(lo, L, index, depth2) :: Tuple{Int, Int}

    for iter in 1:depth2-1
        lo, L = evolver(index, L, lo)
    end

    return lo, L
end


function get_range(L, index , depth1, depth2) :: Tuple{Int, Int, Bool}
    lo, L, dir = get_range_part1(L, index, depth1)
    lo, L = get_range_part2(lo, L, index, depth2)
    return lo, L, dir
end

function kernel(c, depth1, depth2)
    index = (blockDim().x * (blockIdx().x - 1) ) + threadIdx().x - 1
    if index >= length(c)
        return
    end

    lo, n, dir = get_range(length(c), index, depth1, depth2)

    if lo < 0 || n < 0
        return
    end

    m = gp2lt(n)
    if  lo <= index < lo + n - m
        i, j = index, index + m
        @inbounds compare(c, i, j, dir)
    end

    return
end

function kernel_small(c :: AbstractArray{T}, depth1, d2_0, d2_f) where T
    index = (blockDim().x * (blockIdx().x - 1) ) + threadIdx().x - 1
    lo, n, dir = get_range_part1(length(c), index, depth1)
    lo, n = get_range_part2(lo, n, index, d2_0)
    for depth2 in d2_0:d2_f

        if index < length(c) && lo >= 0
            m = gp2lt(n)
            if  lo <= index < lo + n - m
                i, j = index, index + m
                @inbounds compare(c, i, j, dir)
            end
        end
        sync_threads()
        lo, n = evolver(index, n, lo)
    end

    return
end


function bitosort(c)
    log_k0 = c |> length |> log2 |> ceil |> Int

    block_size = 1024
    log_block = block_size |> log2 |> Int

    for log_k in log_k0:-1:1

        j_final = (1+log_k0-log_k)
        for log_j in 1:j_final
            if log_k0 - log_k - log_j + 2 <= log_block
                @cuda blocks=cld(length(c), block_size) threads=block_size kernel_small(c, log_k, log_j, j_final)
                break
            else
                @cuda blocks=cld(length(c), block_size) threads=block_size kernel(c, log_k, log_j)
            end

        end

    end
    synchronize()
end
