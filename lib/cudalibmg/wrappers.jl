
mutable struct CudaLibMGDescriptor
    desc::cudaLibMgMatrixDesc_t

    function CudaLibMGDescriptor(a, grid; rowblocks = size(a, 1), colblocks = size(a, 2), elta = eltype(a) )
        desc = Ref{cudaLibMgMatrixDesc_t}()
        try
            cudaLibMgCreateMatrixDesc(desc, size(a, 1), size(a, 2), rowblocks, colblocks, cudaDataType(elta), grid)
        catch e
            println("size(A) = $(size(a)), rowblocks = $rowblocks, colblocks = $colblocks")
            flush(stdout)
            throw(e)
        end
        return new(desc[])
    end
end

Base.cconvert(::Type{cudaLibMgMatrixDesc_t}, obj::CudaLibMGDescriptor) = obj.desc

mutable struct CudaLibMGGrid
    desc::Ref{cudaLibMgGrid_t}

    function CudaLibMGGrid(num_row_devs, num_col_devs, deviceIds, mapping)
        desc = Ref{cudaLibMgGrid_t}()
        cudaLibMgCreateDeviceGrid(desc, num_row_devs, num_col_devs, deviceIds, mapping)
        return new(desc)
    end
end

Base.cconvert(::Type{cudaLibMgGrid_t}, obj::CudaLibMGGrid) = obj.desc[]

function allocateBuffers(n_row_devs, n_col_devs, num_devices::Int, deviceIdsGrid, descr, mat::Matrix)
    mat_row_block_size = div(size(mat, 1), n_row_devs)
    mat_col_block_size = div(size(mat, 2), n_col_devs)
    mat_buffers  = Vector{CuPtr{Cvoid}}(undef, num_devices)
    mat_numRows  = Vector{Int64}(undef, num_devices)
    mat_numCols  = Vector{Int64}(undef, num_devices)
    streams      = Vector{CuStream}(undef, num_devices)
    typesize = sizeof(eltype(mat))
    ldas = Vector{Int64}(undef, num_devices)
    mat_cpu_bufs = Vector{Matrix{eltype(mat)}}(undef, num_devices)
    for (di, dev) in enumerate(deviceIdsGrid)
        ldas[di]    = mat_col_block_size 
        dev_row     = mod(di - 1, n_row_devs) + 1
        dev_col     = div(di - 1, n_row_devs) + 1

        mat_row_inds     = ((dev_row-1)*mat_row_block_size+1):min(dev_row*mat_row_block_size, size(mat, 1))
        mat_col_inds     = ((dev_col-1)*mat_col_block_size+1):min(dev_col*mat_col_block_size, size(mat, 2))
        mat_cpu_bufs[di] = Array(mat[mat_row_inds, mat_col_inds])
    end
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        if !isassigned(streams, di)
            streams[di] = CuStream()
        end
        mat_gpu_buf = CuMatrix{eltype(mat)}(undef, size(mat))
        unsafe_copyto!(pointer(mat_gpu_buf), pointer(mat_cpu_bufs[di]), length(mat_cpu_bufs[di]), stream = streams[di], async = true)
        mat_buffers[di] = convert(CuPtr{Cvoid}, pointer(mat_gpu_buf))
    end
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        synchronize(streams[di])
    end
    device!(deviceIdsGrid[1])
    return mat_buffers
end

function returnBuffers(n_row_devs, n_col_devs, num_devices::Int, deviceIdsGrid, row_block_size, col_block_size, desc, dDs, D)
    row_block_size = div(size(D, 1), n_row_devs)
    col_block_size = div(size(D, 2), n_col_devs)
    numRows  = [row_block_size for dev in 1:num_devices]
    numCols  = [col_block_size for dev in 1:num_devices]
    typesize = sizeof(eltype(D))
    current_dev = device()
    streams  = Vector{CuStream}(undef, num_devices)
    cpu_bufs = Vector{Matrix{eltype(D)}}(undef, num_devices)
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        if !isassigned(streams, di)
            streams[di] = CuStream()
        end
        dev_row = mod(di - 1, n_row_devs) + 1
        dev_col = div(di - 1, n_row_devs) + 1
        row_inds = ((dev_row-1)*row_block_size+1):min(dev_row*row_block_size, size(D, 1))
        col_inds = ((dev_col-1)*col_block_size+1):min(dev_col*col_block_size, size(D, 2))
        cpu_bufs[di] = Matrix{eltype(D)}(undef, length(row_inds), length(col_inds))
        unsafe_copyto!(pointer(cpu_bufs[di]), convert(CuPtr{eltype(D)}, dDs[di]), length(cpu_bufs[di]), stream = streams[di], async = true)
    end
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        synchronize(streams[di])
        dev_row = mod(di - 1, n_row_devs) + 1
        dev_col = div(di - 1, n_row_devs) + 1
        row_inds = ((dev_row-1)*row_block_size+1):min(dev_row*row_block_size, size(D, 1))
        col_inds = ((dev_col-1)*col_block_size+1):min(dev_col*col_block_size, size(D, 2))
        D[row_inds, col_inds] = cpu_bufs[di]
    end
    device!(deviceIdsGrid[1])
    return D
end

