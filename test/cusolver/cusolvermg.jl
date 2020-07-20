using LinearAlgebra, Test
using CUDA
using CUDA.CUSOLVER

import LinearAlgebra: BlasInt

@testset "CUSOLVERMG" begin

m = 256
n = 512
devs = collect(devices())

if CUDA.has_cusolvermg()
    CUSOLVER.cusolverMgDeviceSelect(CUSOLVER.mg_handle(), length(devs), devs)
    @testset "mg_syevd!" begin
        @testset "element type $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            GC.enable(false)
            A = rand(elty, m, m)
            A += A'
            hW = eigvals(Hermitian(A))
            hV = eigvecs(Hermitian(A))
            W, A = CUSOLVER.mg_syevd!('V','L',A, devs=devs)
            # compare
            @test W ≈ hW 
            @test A*diagm(0=>W)*A' ≈ hV*diagm(0=>hW)*hV'

            A = rand(elty, m, m)
            A += A'
            hW = eigvals(Hermitian(A))
            W = CUSOLVER.mg_syevd!('N','L',A, devs=devs)
            # compare
            @test W ≈ hW 

            GC.enable(true)
        end
    end # elty
    if CUDA.toolkit_version() >= v"11.0"
        @testset "mg_potrf!" begin
            @testset "element type $elty" for elty in [Float64, ComplexF64]#[Float32, Float64, ComplexF32, ComplexF64]
                GC.enable(false)
                A = rand(elty, m, m)
                A = A*A'
                hA = copy(A)
                A = CUSOLVER.mg_potrf!('L',A, devs=devs)
                LinearAlgebra.LAPACK.potrf!('L', hA)
                # compare
                @test A ≈ hA 
                GC.enable(true)
            end
        end # elty

        # sometimes broken for F32???
        @testset "mg_potrf and mg_potri!" begin
            @testset "element type $elty" for elty in [Float64, ComplexF64]#[Float32, Float64, ComplexF32, ComplexF64]
                GC.enable(false)
                A = rand(elty, m, m)
                A = A*A'
                hA = copy(A)
                A = CUSOLVER.mg_potrf!('L',A, devs=devs)
                A = CUSOLVER.mg_potri!('L',A, devs=devs)
                LinearAlgebra.LAPACK.potrf!('L', hA)
                LinearAlgebra.LAPACK.potri!('L', hA)
                # compare
                @test A ≈ hA rtol=1e-1 
                GC.enable(true)
            end
        end # elty

        # sometimes broken for F32???
        @testset "mg_potrf and mg_potrs!" begin
            @testset "element type $elty" for elty in [Float64, ComplexF64]#[Float32, Float64, ComplexF32, ComplexF64]
                GC.enable(false)
                A = rand(elty, m, m)
                B = rand(elty, m, m)
                A = A*A'
                hA = copy(A)
                hB = copy(B)
                LinearAlgebra.LAPACK.potrf!('L', hA)
                LinearAlgebra.LAPACK.potrs!('L', hA, hB)
                A = CUSOLVER.mg_potrf!('L',A, devs=devs)
                B = CUSOLVER.mg_potrs!('L',A,B, devs=devs)
                # compare
                @test A ≈ hA 
                @test B ≈ hB rtol=1e-6
                GC.enable(true)
            end
        end # elty
    end
    if CUDA.toolkit_version() >= v"10.2"
        @testset "getrf!" begin
            @testset "element type $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
                GC.enable(false)
                A      = rand(elty,m,m)
                h_A    = copy(A)
                A,ipiv = CUSOLVER.mg_getrf!(A, devs=devs)
                alu    = LinearAlgebra.LU(A, convert(Vector{BlasInt},ipiv), zero(BlasInt))
                @test h_A ≈ Array(alu)
                GC.enable(true)
            end
        end

        @testset "getrs!" begin
            @testset "element type $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
                GC.enable(false)
                A      = rand(elty,m,m)
                h_A    = copy(A)
                A,ipiv = CUSOLVER.mg_getrf!(A, devs=devs)
                B      = rand(elty, m, div(m,2))
                h_B    = copy(B)
                B      = CUSOLVER.mg_getrs!('N', A, ipiv, B, devs=devs)
                alu    = LinearAlgebra.LU(A, convert(Vector{BlasInt},ipiv), zero(BlasInt))
                @test B ≈ h_A\h_B 
                GC.enable(true)
            end
        end
    end
end
end # cusolvermg testset
