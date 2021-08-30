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

exchange(A, i, j) = begin A[i + 1], A[j + 1] = A[j + 1], A[i + 1] end

compare(A, i, j, dir :: Bool) = if dir == (A[i + 1] > A[j + 1]) exchange(A, i, j) end


function get_range(L, index , depth1, depth2) :: Tuple{Int, Int, Bool}
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

    for iter in 1:depth2-1
        if L <= 1
            return -1, -1, false
        end
        m = gp2lt(L)
        if index < lo + m
            L = m
        else
            lo = lo + m
            L = L - m
        end
    end

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
        compare(c, i, j, dir)
    end

    return
end


function bitosort(c)
    log_k0 = c |> length |> log2 |> ceil |> Int

    for log_k in log_k0:-1:1

        for log_j in 1:(1+log_k0-log_k)
            #println((log_k, log_j))
            @cuda blocks=cld(length(c), 256) threads=256 kernel(c, log_k, log_j )
            #
        end

    end
    synchronize()
end
