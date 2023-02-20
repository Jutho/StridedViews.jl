using Test
using LinearAlgebra
using Random
using StridedViews

Random.seed!(1234)

@testset "construction of StridedView" begin
    @testset for T1 in (Float32, Float64, Complex{Float32}, Complex{Float64})
        A1 = randn(T1, (60, 60))
        B1 = StridedView(A1)
        C1 = StridedView(B1)
        @test C1 === B1
        @test parent(B1) === A1
        @test Base.elsize(B1) == Base.elsize(A1)
        for op1 in (identity, conj, transpose, adjoint)
            if op1 == transpose || op1 == adjoint
                @test op1(A1) == op1(B1) == StridedView(op1(A1))
            else
                @test op1(A1) == op1(B1)
            end
            for op2 in (identity, conj, transpose, adjoint)
                @test op2(op1(A1)) == op2(op1(B1))
            end
        end

        A2 = view(A1, 1:36, 1:20)
        B2 = StridedView(A2)
        for op1 in (identity, conj, transpose, adjoint)
            if op1 == transpose || op1 == adjoint
                @test op1(A2) == op1(B2) == StridedView(op1(A2))
            else
                @test op1(A2) == op1(B2)
            end
            for op2 in (identity, conj, transpose, adjoint)
                @test op2(op1(A2)) == op2(op1(B2))
            end
        end

        A3 = reshape(A1, 360, 10)
        B3 = StridedView(A3)
        @test size(A3) == size(B3)
        @test strides(A3) == strides(B3)
        @test stride(A3, 1) == stride(B3, 1)
        @test stride(A3, 2) == stride(B3, 2)
        @test stride(A3, 3) == stride(B3, 3)
        for op1 in (identity, conj, transpose, adjoint)
            if op1 == transpose || op1 == adjoint
                @test op1(A3) == op1(B3) == StridedView(op1(A3))
            else
                @test op1(A3) == op1(B3)
            end
            for op2 in (identity, conj, transpose, adjoint)
                @test op2(op1(A3)) == op2(op1(B3))
            end
        end

        A4 = reshape(view(A1, 1:36, 1:20), (6, 6, 5, 4))
        B4 = StridedView(A4)
        B4[2,2,2,2] = 3
        @test A4[2,2,2,2] == 3
        for op1 in (identity, conj)
            @test op1(A4) == op1(B4)
            for op2 in (identity, conj)
                @test op2(op1(A4)) == op2(op1(B4))
            end
        end

        A5 = PermutedDimsArray(reshape(view(A1, 1:36, 1:20), (6, 6, 5, 4)), (3, 1, 2, 4))
        B5 = StridedView(A5)
        for op1 in (identity, conj)
            @test op1(A5) == op1(B5)
            for op2 in (identity, conj)
                @test op2(op1(A5)) == op2(op1(B5))
            end
        end

        A6 = reshape(view(A1, 1:36, 1:20), (6, 120))
        @test_throws StridedViews.ReshapeException StridedView(A6)
        try
            StridedView(A6)
        catch ex
            println("Printing error message:")
            show(ex)
            println("")
        end

        # Array with Array elements
        A7 = [randn(T1, (5, 5)) for i in 1:5, j in 1:5]
        B7 = StridedView(A7)
        for op1 in (identity, conj, transpose, adjoint)
            @test op1(A7) == op1(B7) == StridedView(op1(A7))
            for op2 in (identity, conj, transpose, adjoint)
                @test op2(op1(A7)) == op2(op1(B7))
            end
        end

        # Zero-dimensional array
        A8 = randn(T1, ())
        B8 = StridedView(A8)
        @test stride(B8, 1) == stride(B8, 5) == 1
        for op1 in (identity, conj)
            @test op1(A8) == op1(B8) == StridedView(op1(A8))
            for op2 in (identity, conj)
                @test op2(op1(A8)) == op2(op1(B8))
            end
        end
        @test reshape(B8, (1, 1, 1)) == reshape(A8, (1, 1, 1)) ==
              StridedView(reshape(A8, (1, 1, 1))) == sreshape(A8, (1, 1, 1))
        @test reshape(B8, ()) == reshape(A8, ())
    end
end

@testset "elementwise conj, transpose and adjoint" begin
    @testset for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
        A = [rand(T) for i in 1:5, b in 1:4, c in 1:3, d in 1:2]
        B = StridedView(A)

        @test conj(B) == conj(A)
        @test conj(B) == map(conj, B)
        @test map(transpose, B) == map(transpose, A)
        @test map(adjoint, B) == map(adjoint, A)
    end
    @testset for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
        A = [randn(T, (3, 3)) for i in 1:5, b in 1:4, c in 1:3, d in 1:2]
        B = StridedView(A)
        @test Base.elsize(B) == Base.elsize(A)

        @test conj(B) == conj(A)
        @test conj(B) == map(conj, B)
        @test map(transpose, B) == map(transpose, A)
        @test map(adjoint, B) == map(adjoint, A)
    end
end

@testset "reshape and permutedims with StridedView" begin
    @testset for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
        A0 = randn(T, 10)
        @test permutedims(StridedView(A0), (1,)) == A0

        @testset "reshape and permutedims with $N-dimensional arrays" for N in 2:6
            let dims = ntuple(n -> rand(1:div(60, N)), N)
                A = rand(T, dims)
                B = StridedView(A)
                @test conj(A) == conj(B)
                p = randperm(N)
                B2 = permutedims(B, p)
                A2 = permutedims(A, p)
                @test B2 == A2
            end

            let dims = ntuple(n -> 10, N)
                A = rand(T, dims)
                B = StridedView(A)
                @test conj(A) == conj(B)
                p = randperm(N)
                B2 = permutedims(B, p)
                A2 = permutedims(A, p)
                @test B2 == A2

                B2 = sreshape(B, (2, 5, ntuple(n -> 10, N - 2)..., 5, 2))
                A2 = sreshape(A, (2, 5, ntuple(n -> 10, N - 2)..., 5, 2)...)
                A3 = reshape(A, size(A2))
                @test B2 == A3
                @test B2 == A2
                p = randperm(N + 2)
                @test conj(permutedims(B2, p)) == conj(permutedims(A3, p))
            end
        end

        @testset "more reshape" begin
            A = randn(4, 0)
            B = StridedView(A)
            @test_throws DimensionMismatch sreshape(B, (4, 1))
            C = sreshape(B, (2, 1, 2, 0, 1))
            @test sreshape(C, (4, 0)) == A

            A = randn(4, 1, 2)
            B = StridedView(A)
            @test_throws DimensionMismatch sreshape(B, (4, 4))
            C = sreshape(B, (2, 1, 1, 4, 1, 1))
            @test C == reshape(A, (2, 1, 1, 4, 1, 1))
            @test sreshape(C, (4, 1, 2)) == A
        end
    end
end

@testset "views with StridedView" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = randn(T, (10, 10, 10, 10))
        B = StridedView(A)
        @test isa(view(B, :, 1:5, 3, 1:5), StridedView)
        @test isa(view(B, :, [1, 2, 3], 3, 1:5), Base.SubArray)
        @test isa(sview(B, :, 1:5, 3, 1:5), StridedView)
        @test_throws MethodError sview(B, :, [1, 2, 3], 3, 1:5)

        @test view(A, 1:38) == view(B, 1:38) == sview(A, 1:38) == sview(B, 1:38)

        @test view(B, :, 1:5, 3, 1:5) == view(A, :, 1:5, 3, 1:5) == sview(A, :, 1:5, 3, 1:5)
        @test view(B, :, 1:5, 3, 1:5) === sview(B, :, 1:5, 3, 1:5) === B[:, 1:5, 3, 1:5]
        @test view(B, :, 1:5, 3, 1:5) == StridedView(view(A, :, 1:5, 3, 1:5))
        @test pointer(view(B, :, 1:5, 3, 1:5)) ==
              pointer(StridedView(view(A, :, 1:5, 3, 1:5)))
        @test StridedViews.offset(view(B, :, 1:5, 3, 1:5)) == 2 * stride(B, 3)
    end
end

using Aqua
Aqua.test_all(StridedViews)