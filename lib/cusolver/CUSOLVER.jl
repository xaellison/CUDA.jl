module CUSOLVER

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType, cudaDataType
using ..CUDA: libcusolver, libcusolvermg, @allowscalar, assertscalar, unsafe_free!, @retry_reclaim

using ..CUBLAS: cublasFillMode_t, cublasOperation_t, cublasSideMode_t, cublasDiagType_t
using ..CUSPARSE: cusparseMatDescr_t
using ..CUDALIBMG: allocateBuffers, returnBuffers

using CEnum

# core library
include("libcusolver_common.jl")
include("error.jl")
include("libcusolver.jl")

# low-level wrappers
include("util.jl")
include("wrappers.jl")

# high-level integrations
include("linalg.jl")

# thread cache for task-local library handles
const thread_dense_handles  = Vector{Union{Nothing,cusolverDnHandle_t}}()
const thread_sparse_handles = Vector{Union{Nothing,cusolverSpHandle_t}}()
const thread_mg_handles  = Vector{Union{Nothing,cusolverMgHandle_t}}()
const thread_mg_grids    = Vector{Union{Nothing,cudaLibMgGrid_t}}()

function dense_handle()
    tid = Threads.threadid()
    if @inbounds thread_dense_handles[tid] === nothing
        ctx = context()
        thread_dense_handles[tid] = get!(task_local_storage(), (:CUSOLVER, :dense, ctx)) do
            handle = cusolverDnCreate()
            finalizer(current_task()) do task
                CUDA.isvalid(ctx) || return
                context!(ctx) do
                    cusolverDnDestroy(handle)
                end
            end

            handle
        end
    end
    @inbounds thread_dense_handles[tid]
end

function sparse_handle()
    tid = Threads.threadid()
    if @inbounds thread_sparse_handles[tid] === nothing
        ctx = context()
        thread_sparse_handles[tid] = get!(task_local_storage(), (:CUSOLVER, :sparse, ctx)) do
            handle = cusolverSpCreate()
            finalizer(current_task()) do task
                CUDA.isvalid(ctx) || return
                context!(ctx) do
                    cusolverSpDestroy(handle)
                end
            end

            handle
        end
    end
    @inbounds thread_sparse_handles[tid]
end

function mg_handle()
    tid = Threads.threadid()
    if @inbounds thread_mg_handles[tid] === nothing
        ctx = context()
        thread_mg_handles[tid] = get!(task_local_storage(), (:CUSOLVER, :mg, ctx)) do
            handle = cusolverMgCreate()
            finalizer(current_task()) do task
                CUDA.isvalid(ctx) || return
                context!(ctx) do
                    cusolverMgDestroy(handle)
                end
            end

            handle
        end
    end
    @inbounds thread_mg_handles[tid]
end

function __init__()
    resize!(thread_dense_handles, Threads.nthreads())
    fill!(thread_dense_handles, nothing)

    resize!(thread_sparse_handles, Threads.nthreads())
    fill!(thread_sparse_handles, nothing)

    resize!(thread_mg_handles, Threads.nthreads())
    fill!(thread_mg_handles, nothing)

    resize!(thread_mg_grids, Threads.nthreads())
    fill!(thread_mg_grids, nothing)

    CUDA.atcontextswitch() do tid, ctx
        thread_dense_handles[tid] = nothing
        thread_sparse_handles[tid] = nothing
        thread_mg_handles[tid] = nothing
        thread_mg_grids[tid] = nothing
    end

    CUDA.attaskswitch() do tid, task
        thread_dense_handles[tid] = nothing
        thread_sparse_handles[tid] = nothing
        thread_mg_handles[tid] = nothing
        thread_mg_grids[tid] = nothing
    end
end

end
