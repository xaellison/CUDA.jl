
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


function pseudo_kernel(c, depth1, depth2)
    @info Set(map(i->get_range(length(c), i, depth1, depth2), 0:length(c)-1))
    for index in 0:length(c)-1
        lo, n, dir = get_range(length(c), index, depth1, depth2)
        if lo < 0
            continue
        end
        #println((lo, n, dir))
        m = gp2lt(n)
        #@assert lo <= index <= lo + n - m -1
        #for i in lo:(lo+n-m-1)
        if  lo <= index < lo + n - m
        #    println("~")
            compare(c, index, index+m, dir)
        end
    end
end


function bitosort(c)
    log_k0 = c |> length |> log2 |> ceil |> Int

    for log_k in log_k0:-1:1

        for log_j in 1:(1+log_k0-log_k)
            println((log_k, log_j))
            pseudo_kernel(c, log_k, log_j )
        end

    end

end
