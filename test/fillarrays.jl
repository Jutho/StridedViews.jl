module FillArrayTests

using Test
using LinearAlgebra
using StridedViews
using FillArrays
using Random
Random.seed!(1234)

@testset "FillArrays" verbose = true begin
    @testset for T1 in (Float32, Float64, Complex{Float32}, Complex{Float64})
        A1 = Fill(rand(T1), (60, 60))
        B1 = StridedView(A1)
        C1 = StridedView(B1)
        @test C1 === B1
        @test parent(B1) === A1
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

        A5 = PermutedDimsArray(reshape(view(A1, 1:36, 1:20), (6, 6, 5, 4)), (3, 1, 2, 4))
        B5 = StridedView(A5)
        for op1 in (identity, conj)
            @test op1(A5) == op1(B5)
            for op2 in (identity, conj)
                @test op2(op1(A5)) == op2(op1(B5))
            end
        end

        # Zero-dimensional array is currently broken, see https://github.com/JuliaArrays/FillArrays.jl/issues/145
        # A8 = Fill(rand(T1), ())
        # B8 = StridedView(A8)
        # @test stride(B8, 1) == stride(B8, 5) == 1
        # for op1 in (identity, conj)
        #     @test op1(A8) == op1(B8) == StridedView(op1(A8))
        #     for op2 in (identity, conj)
        #         @test op2(op1(A8)) == op2(op1(B8))
        #     end
        # end
        # @test reshape(B8, (1, 1, 1)) == reshape(A8, (1, 1, 1)) ==
        #       StridedView(reshape(A8, (1, 1, 1))) == sreshape(A8, (1, 1, 1))
        # @test reshape(B8, ()) == reshape(A8, ())
    end

    @testset "transpose and adjoint with vector StridedView" begin
        @testset for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
            A = Fill(rand(T), (60,))

            @test sreshape(transpose(A), (1, length(A))) == transpose(A)
            @test sreshape(adjoint(A), (1, length(A))) == adjoint(A)
        end
    end

    @testset "reshape and permutedims with StridedView" begin
        @testset for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
            @testset "reshape and permutedims with $N-dimensional arrays" for N in 2:6
                let dims = ntuple(n -> rand(1:div(60, N)), N)
                    A = Fill(rand(T), dims)
                    B = StridedView(A)
                    @test conj(A) == conj(B)
                    p = randperm(N)
                    B2 = permutedims(B, p)
                    A2 = permutedims(A, p)
                    @test B2 == A2
                end

                let dims = ntuple(n -> 10, N)
                    A = Fill(rand(T), dims)
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
                A = Ones(4, 0)
                B = StridedView(A)
                @test_throws DimensionMismatch sreshape(B, (4, 1))
                C = sreshape(B, (2, 1, 2, 0, 1))
                @test sreshape(C, (4, 0)) == A

                A = Trues(4, 1, 2)
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
            A = Fill(rand(T), (10, 10, 10, 10))
            B = StridedView(A)
            @test isa(view(B, :, 1:5, 3, 1:5), StridedView)
            @test isa(view(B, :, [1, 2, 3], 3, 1:5), Base.SubArray)
            @test isa(sview(B, :, 1:5, 3, 1:5), StridedView)
            @test_throws MethodError sview(B, :, [1, 2, 3], 3, 1:5)

            @test view(A, 1:38) == view(B, 1:38) == sview(A, 1:38) == sview(B, 1:38)

            @test view(B, :, 1:5, 3, 1:5) == view(A, :, 1:5, 3, 1:5) ==
                  sview(A, :, 1:5, 3, 1:5)
            @test view(B, :, 1:5, 3, 1:5) === sview(B, :, 1:5, 3, 1:5) === B[:, 1:5, 3, 1:5]
            @test view(B, :, 1:5, 3, 1:5) == StridedView(view(A, :, 1:5, 3, 1:5))
            @test StridedViews.offset(view(B, :, 1:5, 3, 1:5)) == 2 * stride(B, 3)
        end
    end
end

end
