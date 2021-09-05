using CUDA
"""
@inline function gp2lt(x :: Int)
   x |= x >> 1
   x |= x >> 2
   x |= x >> 4
   x |= x >> 8
   x |= x >> 16
   x |= x >> 32
   xor(x, x >> 1)
end
"""

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

gp2gt(n) = if n <= 1 0 else Int(2^ceil(log2(n))) end

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

    lo, n, dir = get_range(length(c), index, depth1, depth2)

    if ! (lo < 0 || n < 0) && !(index >= length(c))

        m = gp2lt(n)
        if  lo <= index < lo + n - m
            i, j = index, index + m
            @inbounds compare(c, i, j, dir)
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

function kernel_small(c :: AbstractArray{T}, depth1, d2_0, d2_f, debug) where T
    _lo, _n, dir = blockit(length(c), blockIdx().x -1, depth1, d2_0)
    grid_index = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    index = _lo + threadIdx().x - 1
    if threadIdx().x <= _n && index <= length(debug)
    #    debug[grid_index] = index #(_lo, _n, dir)
    end
    lo, n = _lo, _n

    for depth2 in d2_0:d2_f
        lo, n, dir = get_range(length(c), index, depth1, depth2)

        if ! (lo < 0 || n < 0) && !(index >= length(c)) && threadIdx().x <= _n && _lo >= 0

            m = gp2lt(n)
            if  lo <= index < lo + n - m
                i, j = index, index + m
                @inbounds compare(c, i, j, dir)
            end
        end

        sync_threads()
    end

    return
end


function bitosort(c, block_size=1024)
    log_k0 = c |> length |> log2 |> ceil |> Int
    debug = CuArray(zeros(Int, length(c)))
    @info block_size
    log_block = block_size |> log2 |> Int

    for log_k in log_k0:-1:1

        j_final = (1+log_k0-log_k)

        for log_j in 1:j_final
            if log_k0 - log_k - log_j + 2 < log_block

                _block_size = 1 << abs(j_final + 1 - log_j)
                b= gp2gt(cld(length(c), _block_size))
                @info "smol $log_k, $log_j, $j_final. blocks = $b"
                @cuda blocks=b threads=_block_size kernel_small(c, log_k, log_j, j_final, debug)
                synchronize()

                #println(debug)

                println(length(filter(x->x!=0, Array(debug))))
                println(length(Set(Array(debug))))
                println()
                #return
                break
            else
                @cuda blocks=cld(length(c), block_size) threads=block_size kernel(c, log_k, log_j)
            end

        end

    end
    synchronize()
end
