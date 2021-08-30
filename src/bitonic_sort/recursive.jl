# https://www.inf.hs-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm

gp2lt(n) = if n <= 1 0 else Int(2^floor(log2(n - 1))) end
gp2gt(n) = if n <= 1 0 else Int(2^ceil(log2(n))) end

exchange(A, i, j) = begin A[i + 1], A[j + 1] = A[j + 1], A[i + 1] end

compare(A, i, j, dir :: Bool) = if dir == (A[i + 1] > A[j + 1]) exchange(A, i, j) end


function merge(A, lo, n, dir, s_depth, m_depth, logger)
    if n <= 1
        return
    end

    m = gp2lt(n)
    push!(logger, (s_depth, m_depth, length(logger), lo, n, m, dir))
    # println("$(repeat("\t", depth)) M: $lo, $n, $m, $dir")
    # this is where the parallelization can happen
    # determine all the lo's, m's in parallel
    # launch Sum(n-m_i, i) threads aware of their lo, m
    @info "depths: $s_depth $m_depth"
    for i in lo:(lo+n-m-1)
        @info "lo = $lo n = $n dir = $dir"
        compare(A, i, i+m, dir)
    end
    merge(A, lo, m, dir, s_depth, m_depth+1, logger)
    merge(A, lo+m, n-m, dir, s_depth, m_depth+1, logger)
end


function bitosort(A, lo, n, dir, depth=1, logger=[])
    if n <= 1
        return
    end
    m = n÷2

    bitosort(A, lo, m, !dir, depth +1, logger)
    bitosort(A, lo+m, n-m, dir, depth+1, logger)
    merge(A, lo, n, dir, depth, 1, logger)
    logger
end

# this shows that the network "nodes" can be arbitrarily ordered as long as they are first ordered by the outer/inner loop indices.. even ones with weird n/m
function test(n)
    a0 = rand(n)
    a = copy(a0)
    network = bitosort(a, 0, length(a), true)
    # make sure recursive form worked
    @assert a == sort(a0)

    # shuffle within outer/inner indexes
    sort!(network, by=t->(-t[1], t[2], rand()))
    a = copy(a0)
    # manually apply permuted network
    for step in network
        _, _, _, lo, n, m, dir = step
        for i in lo:(lo+n-m-1)
            compare(a, i, i+m, dir)
        end
    end

    # verify correctness
    @assert a == sort(a0)
    return
end
