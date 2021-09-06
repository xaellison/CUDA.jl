using Test

@testset "integer functions" begin

    @test gp2lte(1) == 1
    @test gp2lte(2) == 2
    @test gp2lte(3) == 2
    @test gp2lte(4) == 4
    @test gp2lte(1 << 8 - 1) == 1 << 7
    @test gp2lte(1 << 8) == 1 << 8
    @test gp2lte(1 << 8 + 1) == 1 << 8

    @test gp2lt(1) == 0
    @test gp2lt(2) == 1
    @test gp2lt(3) == 2
    @test gp2lt(4) == 2
    @test gp2lt(5) == 4
    @test gp2lt(1 << 8 - 1) == 1 << 7
    @test gp2lt(1 << 8) == 1 << 7
    @test gp2lt(1 << 8 + 1) == 1 << 8
end


@testset "recursive network" begin

    for n in 64:128
        a0 = rand(n)
        a = copy(a0)
        bitosort(a, 0, length(a), true)
        @test a == sort(a0)
    end

    # this shows that the network "nodes" can be arbitrarily ordered as long as they are first ordered by the outer/inner loop indices.. even ones with weird n/m
    function test_shuffled_network(n)
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
            for i in lo:(lo + n - m - 1)
                compare(a, i, i + m, dir)
            end
        end
        # verify correctness of the permuted sort
        @test a == sort(a0)

        # this tests that we can use loops to produce `expected_loops` and that this will be
        # equivalent to the recursive implementation
        # TODO make this an interator and use here as well as in actual sort
        expected_loops = []
        log_k0 = a |> length |> log2 |> ceil |> Int
        for log_k in log_k0:-1:1

            j_final = (1 + log_k0 - log_k)

            for log_j in 1:j_final
                push!(expected_loops, (log_k, log_j))
            end
        end

        @test sort(collect(Set(map(t->t[1:2], network))), by=t->(-t[1], t[2])) == expected_loops

        return
    end

    for n in 64:128
        test_shuffled_network(n)
    end

end
