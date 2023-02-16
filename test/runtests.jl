using Test
using LinearAlgebra
using Random
using StridedViews

Random.seed!(1234)

@testset "construction of StridedView" begin
    @testset for T1 in (Float32, Float64, Complex{Float32}, Complex{Float64})
        A1 = randn(T1, (60, 60))
        B1 = StridedView(A1)
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
        for op1 in (identity, conj)
            @test op1(A4) == op1(B4)
            for op2 in (identity, conj)
                @test op2(op1(A4)) == op2(op1(B4))
            end
        end

        A5 = reshape(view(A1, 1:36, 1:20), (6, 120))
        @test_throws StridedViews.ReshapeException StridedView(A5)

        A6 = [randn(T1, (5, 5)) for i in 1:5, j in 1:5]

        B6 = StridedView(A6)
        for op1 in (identity, conj, transpose, adjoint)
            @test op1(A6) == op1(B6) == StridedView(op1(A6))
            for op2 in (identity, conj, transpose, adjoint)
                @test op2(op1(A6)) == op2(op1(B6))
            end
        end
    end
end

@testset "elementwise conj, transpose and adjoint" begin
    @testset for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
        A = [randn(T, (3, 3)) for i in 1:5, b in 1:4, c in 1:3, d in 1:2]
        Ac = deepcopy(A)
        B = StridedView(A)

        @test conj(B) == conj(A)
        @test conj(B) == map(conj, B)
        @test map(transpose, B) == map(transpose, A)
        @test map(adjoint, B) == map(adjoint, A)
    end
end

@testset "reshape and permutedims with StridedView" begin
    @testset for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
        A0 = randn(T, 10)
        GC.@preserve A0 begin
            @test permutedims(StridedView(A0), (1,)) == A0
        end

        @testset "in-place matrix operations" begin
            A1 = randn(T, (1000, 1000))
            A2 = similar(A1)
            A1c = copy(A1)
            A2c = copy(A2)
            GC.@preserve A1c A2c begin
                B1 = StridedView(A1c)
                B2 = StridedView(A2c)

                @test conj!(A1) == conj!(B1)
                @test adjoint!(A2, A1) == adjoint!(B2, B1)
                @test transpose!(A2, A1) == transpose!(B2, B1)
                @test permutedims!(A2, A1, (2, 1)) == permutedims!(B2, B1, (2, 1))
            end
        end

        @testset "reshape and permutedims with $N-dimensional arrays" for N in 2:6
            dims = ntuple(n -> rand(1:div(60, N)), N)
            A = rand(T, dims)
            Ac = copy(A)
            GC.@preserve Ac begin
                B = StridedView(Ac)
                @test conj(A) == conj(B)
                p = randperm(N)
                B2 = permutedims(B, p)
                A2 = permutedims(A, p)
                @test B2 == A2
                @test copy(B2) == A2
                @test convert(Array, B2) == A2
            end

            dims = ntuple(n -> 10, N)
            A = rand(T, dims)
            Ac = copy(A)
            GC.@preserve Ac begin
                B = StridedView(Ac)
                @test conj(A) == conj(B)
                p = randperm(N)
                B2 = permutedims(B, p)
                A2 = permutedims(A, p)
                @test B2 == A2
                @test copy(B2) == A2
                @test convert(Array, B2) == A2

                B2 = sreshape(B, (2, 5, ntuple(n -> 10, N - 2)..., 5, 2))
                A2 = reshape(A, (2, 5, ntuple(n -> 10, N - 2)..., 5, 2))
                A3 = reshape(copy(A), size(A2))
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

        @test view(B, :, 1:5, 3, 1:5) == view(A, :, 1:5, 3, 1:5)
        @test view(B, :, 1:5, 3, 1:5) === sview(B, :, 1:5, 3, 1:5) === B[:, 1:5, 3, 1:5]
        @test view(B, :, 1:5, 3, 1:5) == StridedView(view(A, :, 1:5, 3, 1:5))
        @test pointer(view(B, :, 1:5, 3, 1:5)) ==
              pointer(StridedView(view(A, :, 1:5, 3, 1:5)))
        @test StridedViews.offset(view(B, :, 1:5, 3, 1:5)) == 2 * stride(B, 3)
    end
end

using Aqua
Aqua.test_all(StridedViews)