using CUDA

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

gp2gt(n) = if n <= 1 0 else Int(2^ceil(log2(n))) end

@inline exchange(A, i, j) = begin A[i + 1], A[j + 1] = A[j + 1], A[i + 1] end

@inline compare(A, i, j, dir :: Bool, by, lt) = if dir != lt(by(A[i + 1]) , by(A[j + 1])) exchange(A, i, j) end

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

function kernel(c, depth1, depth2, by::F1, lt::F2) where {F1, F2}
    index = (blockDim().x * (blockIdx().x - 1) ) + threadIdx().x - 1

    lo, n, dir = get_range(length(c), index, depth1, depth2)

    if lo <= index <= lo + n

        m = gp2lt(n)
        if  lo <= index < lo + n - m
            i, j = index, index + m
            @inbounds compare(c, i, j, dir, by, lt)
        end
    end
    return
end

function blockit(L, index, depth1, depth2)

    lo = 0
    dir = true
    tmp = index  * 2
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

function kernel_small(c :: AbstractArray{T}, depth1, d2_0, d2_f, by::F1, lt::F2) where {T,F1,F2}
    block_swap = @cuDynamicSharedMem(T, (blockDim().x, blockDim().y))
    virtual_block = (blockIdx().x - 1) * blockDim().y + threadIdx().y - 1
    #virtual_block = (threadIdx().y - 1) * (gridDim().x ) + blockIdx().x - 1
    _lo, _n, dir = blockit(length(c), virtual_block, depth1, d2_0)
    index = _lo + threadIdx().x - 1

    in_range = _lo <= index < _lo + _n && index < length(c)

    swap = view(block_swap, :, threadIdx().y)

    sync_threads()
    if in_range
        swap[threadIdx().x] = c[index + 1]
    end
    sync_threads()
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
    if in_range
        c[index + 1] = swap[threadIdx().x]
    end
    return
end


function bitosort(c, block_size=1024; by=identity, lt=isless)
    log_k0 = c |> length |> log2 |> ceil |> Int
    log_block = block_size |> log2 |> Int

    for log_k in log_k0:-1:1

        j_final = (1+log_k0-log_k)

        for log_j in 1:j_final
            if log_k0 - log_k - log_j + 2 <= log_block

                _block_size = 1 << abs(j_final + 1 - log_j)

                b = max(1, gp2gt(cld(length(c), _block_size)))
                W = warpsize(device())
                if _block_size < W
                    b = b ÷ (W ÷ _block_size)
                    _block_size = (_block_size, W ÷ _block_size)
                else
                    _block_size = (_block_size, 1)
                end

                @cuda blocks=b threads=_block_size shmem=sizeof(eltype(c))*prod(_block_size) kernel_small(c, log_k, log_j, j_final, by, lt)
                synchronize()

                break
            else
                b = max(1, cld(length(c), block_size) )
                @cuda blocks=b threads=block_size kernel(c, log_k, log_j, by, lt)
            end

        end

    end
end
