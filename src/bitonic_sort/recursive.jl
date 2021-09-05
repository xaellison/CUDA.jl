# https://www.inf.hs-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm

@inline function gp2lte(x :: Int) :: Int

   x |= x >> 1
   x |= x >> 2
   x |= x >> 4
   x |= x >> 8
   x |= x >> 16
   x |= x >> 32
   xor(x, x >> 1)
end

@inline function gp2lt(x :: Int) :: Int
    gp2lte(x - 1)
end

# used outside of kernels
gp2gt(n) = if n <= 1 0 else Int(2 ^ ceil(log2(n))) end

exchange(A, i, j) = begin A[i + 1], A[j + 1] = A[j + 1], A[i + 1] end

compare(A, i, j, dir :: Bool) = if dir == (A[i + 1] > A[j + 1]) exchange(A, i, j) end


function merge(A, lo, n, dir, s_depth, m_depth, logger)
    if n <= 1
        return
    end

    m = gp2lt(n)
    push!(logger, (s_depth, m_depth, length(logger), lo, n, m, dir))
    for i in lo:(lo + n - m - 1)
        compare(A, i, i + m, dir)
    end
    merge(A, lo, m, dir, s_depth, m_depth + 1, logger)
    merge(A, lo + m, n - m, dir, s_depth, m_depth + 1, logger)
end


function bitosort(A, lo, n, dir, depth=1, logger=[])
    if n <= 1
        return
    end
    m = n ÷ 2

    bitosort(A, lo, m, !dir, depth + 1, logger)
    bitosort(A, lo + m, n - m, dir, depth + 1, logger)
    merge(A, lo, n, dir, depth, 1, logger)
    logger
end
