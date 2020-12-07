"""
Unit tests for quicksort.
"""

using DataStructures, Random, Test, CUDA.Sorting
import CUDA.Sorting: pow2_floor, Θ, seek_last_less_than,
        partition_batches_kernel, consolidate_batch_partition

@testset "Quicksort Integer Functions" begin
@test pow2_floor(1) == 1
@test pow2_floor(2) == 2
@test pow2_floor(3) == 2
@test pow2_floor(5) == 4
@test pow2_floor(8) == 8

@test Θ(0) == 0
@test Θ(1) == 1
@test Θ(2) == 1

@test seek_last_less_than([1, 2, 2, 3, 4, 1, 2, 2, 3, 4], 3, 0, 5, true) == 3
@test seek_last_less_than([1, 2, 2, 3, 4, 1, 2, 2, 3, 4], 3, 5, 10, true) == 8
end

function test_batch_partition(T, N, lo, hi, seed)
    my_range = lo + 1 : hi
    Random.seed!(seed)
    original = rand(T, N)
    A = CuArray(original)

    pivot = rand(original[my_range])
    block_N, block_dim = -1, -1

    function get_config(kernel)
        get_shmem(threads) = threads * (sizeof(Int32) + max(4, sizeof(T)))

        fun = kernel.fun
        config = launch_configuration(fun, shmem=threads->get_shmem(threads), max_threads=1024)

        threads = pow2_floor(config.threads)
        blocks = ceil(Int, (hi - lo) ./ threads)
        block_N = blocks
        block_dim = threads
        @assert block_dim >= 32 "This test assumes block size can be >= 32"
        shmem = get_shmem(threads)
        return (threads=threads, blocks=blocks, shmem=shmem)
    end
    @cuda config=get_config partition_batches_kernel(A, pivot, lo, hi, Val(true))
    synchronize()

    post_sort = Array(A)

    sort_match = true

    for block in 1:block_N
        block_range = lo + 1 + (block - 1) * block_dim: min(hi, lo + block * block_dim)
        temp = original[block_range]
        #= this shows that batch partitioning is a stable sort where key for
        each value v is whether v > or <= pivot =#
        expected_sort = vcat(filter(x -> x < pivot, temp), filter(x -> x >= pivot, temp))
        sort_match &= post_sort[block_range] == expected_sort
    end

    @test sort_match
end

@testset "Quicksort batch partition" begin
test_batch_partition(Int8, 10000, 2000, 6000, 0)
test_batch_partition(Int8, 10000, 2000, 6000, 1)
test_batch_partition(Int8, 10000000, 0, 10000000, 0)
test_batch_partition(Int8, 10000000, 5000, 500000, 0)
test_batch_partition(Int8, 10000, 0, 10000, 0)
test_batch_partition(Int8, 10000, 2000, 6000, 0)
test_batch_partition(Int8, 10000, 2000, 6000, 1)
test_batch_partition(Int8, 10000000, 0, 10000000, 0)
test_batch_partition(Int8, 10000000, 5000, 500000, 0)

test_batch_partition(Float32, 10000, 0, 10000, 0)
test_batch_partition(Float32, 10000, 2000, 6000, 0)
test_batch_partition(Float32, 10000, 2000, 6000, 1)
test_batch_partition(Float32, 10000000, 0, 10000000, 0)
test_batch_partition(Float32, 10000000, 5000, 500000, 0)
test_batch_partition(Float32, 10000, 0, 10000, 0)
test_batch_partition(Float32, 10000, 2000, 6000, 0)
test_batch_partition(Float32, 10000, 2000, 6000, 1)
test_batch_partition(Float32, 10000000, 0, 10000000, 0)
test_batch_partition(Float32, 10000000, 5000, 500000, 0)
end

function test_consolidate_kernel(vals, pivot, my_floor, L, b_sums, dest, parity)
    i = threadIdx().x
    p = consolidate_batch_partition(vals, pivot, my_floor, L, b_sums, parity)
    if i == 1
        dest[1] = p
    end
    return nothing
end

function test_consolidate_partition(T, N, lo, hi, seed, block_dim)
    #@info "$T, $N, $lo, $hi, $seed; $nothing, $shmem=false"
    # assuming partition_batches works, we can validate consolidate by
    # checking that together they partition a large domain
    my_range = lo + 1 : hi
    Random.seed!(seed)
    original = rand(T, N)
    A = CuArray(original)
    pivot = rand(original[my_range])

    threads = blocks = -1
    sums = CuArray(zeros(Int32, ceil(Int, hi - lo / block_dim)))

    function get_config(kernel)
        get_shmem(threads) = threads * (sizeof(Int32) + max(4, sizeof(T)))
        config = launch_configuration(kernel.fun, shmem=threads->get_shmem(threads), max_threads=1024)

        threads = isnothing(block_dim) ? pow2_floor(config.threads) : block_dim
        blocks = ceil(Int, (hi - lo) ./ threads)

        shmem = get_shmem(threads)
        return (threads=threads, blocks=blocks, shmem=shmem)
    end

    @cuda config=get_config partition_batches_kernel(A, pivot, lo, hi, Val(true))
    synchronize()
    dest = CuArray(zeros(Int32, 1))

    @cuda threads=threads test_consolidate_kernel(A, pivot, lo, hi - lo, sums, dest, true)
    synchronize()

    partition = Array(dest)[1]
    temp = original[my_range]
    post_sort = Array(A)
    #= consolidation is a highly unstable sort (again, by pivot comparison as
    the key) so we compare by counting each element =#
    cc(x) = x |> counter |> collect |> sort
    @test cc(original) == cc(post_sort)
    @test all(post_sort[lo + 1 : partition] |> cc .== filter(x -> x < pivot, temp) |> cc)
    @test all(post_sort[partition + 1 : hi] |> cc .== filter(x -> x >= pivot, temp) |> cc)
end

@testset "Quicksort consolidate partition" begin
test_consolidate_partition(Int8, 10000, 0, 10000, 0, 16)
test_consolidate_partition(Int8, 10000, 0, 10000, 0, 32)
test_consolidate_partition(Int8, 10000, 0, 10000, 0, 64)
test_consolidate_partition(Int8, 10000, 9, 6333, 0, 16)
test_consolidate_partition(Int8, 10000, 9, 6333, 0, 32)
test_consolidate_partition(Int8, 10000, 9, 6333, 0, 64)
test_consolidate_partition(Int8, 10000, 129, 9999, 0, 16)
test_consolidate_partition(Int8, 10000, 129, 9999, 0, 32)
test_consolidate_partition(Int8, 10000, 129, 9999, 0, 64)
test_consolidate_partition(Int8, 10000, 0, 10000, 1, 16)
test_consolidate_partition(Int8, 10000, 0, 10000, 2, 32)
test_consolidate_partition(Int8, 10000, 0, 10000, 3, 64)
test_consolidate_partition(Int8, 10000, 9, 6333, 4, 16)
test_consolidate_partition(Int8, 10000, 9, 6333, 5, 32)
test_consolidate_partition(Int8, 10000, 9, 6333, 6, 64)
test_consolidate_partition(Int8, 10000, 129, 9999, 7, 16)
test_consolidate_partition(Int8, 10000, 129, 9999, 8, 32)
test_consolidate_partition(Int8, 10000, 129, 9999, 9, 64)
test_consolidate_partition(Int8, 10000, 3329, 9999, 10, 16)
test_consolidate_partition(Int8, 10000, 3329, 9999, 11, 32)
test_consolidate_partition(Int8, 10000, 3329, 9999, 12, 64)

end

function test_quicksort(T, f, N)
    a = map(x -> T(f(x)), 1:N)
    c = CuArray(a)
    # if sort! fell back to Base implementation, CUDA would error over
    # disallowed scalar indexing
    CUDA.allowscalar(false)
    sort!(c)
    @test Array(c) == sort(a)
end


@testset "Quicksort" begin
# test pre-sorted
test_quicksort(Int, x -> x, 1000000)
test_quicksort(Int32, x -> x, 1000000)
test_quicksort(Float64, x -> x, 1000000)
test_quicksort(Float32, x -> x, 1000000)

# test reverse sorted
test_quicksort(Int, x -> -x, 1000000)
test_quicksort(Float32, x -> -x, 1000000)

# test random arrays
Random.seed!(0)

test_quicksort(Int, x -> rand(Int), 10000)
test_quicksort(Int32, x -> rand(Int32), 10000)
test_quicksort(Int8, x -> rand(Int8), 10000)
test_quicksort(Float64, x -> rand(Float64), 10000)
test_quicksort(Float32, x -> rand(Float32), 10000)
test_quicksort(Float16, x -> rand(Float16), 10000)

# test where more copies of each value than fit in one block
test_quicksort(Int8, x -> rand(Int8), 4000000)

# test various sync depths
CUDA.limit!(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH, 0)
test_quicksort(Int, x -> rand(Int), 100000)
CUDA.limit!(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH, 1)
test_quicksort(Int, x -> rand(Int), 100000)
CUDA.limit!(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH, 2)
test_quicksort(Int, x -> rand(Int), 100000)
CUDA.limit!(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH, 3)
test_quicksort(Int, x -> rand(Int), 100000)
CUDA.limit!(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH, 4)
test_quicksort(Int, x -> rand(Int), 100000)
end
