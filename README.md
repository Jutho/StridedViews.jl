# StridedViews

| **Build Status** | **Coverage** | **Quality assurance** |
|:----------------:|:------------:|:---------------------:|
| [![CI][ci-img]][ci-url] [![CI (Julia nightly)][ci-julia-nightly-img]][ci-julia-nightly-url] | [![Codecov][codecov-img]][codecov-url] | [![Aqua QA][aqua-img]][aqua-url] |

[ci-img]: https://github.com/Jutho/StridedViews.jl/actions/workflows/CI.yml/badge.svg?branch=main
[ci-url]: https://github.com/Jutho/StridedViews.jl/actions/workflows/CI.yml?query=branch%3Amain

[ci-julia-nightly-img]: https://github.com/Jutho/StridedViews.jl/workflows/CI%20(Julia%20nightly)/badge.svg
[ci-julia-nightly-url]: https://github.com/Jutho/StridedViews.jl/actions?query=workflow%3A%22CI+%28Julia+nightly%29%22

[codecov-img]: https://codecov.io/gh/Jutho/StridedViews.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/Jutho/StridedViews.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

<!-- [genie-img]: https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/Strided
[genie-url]: https://pkgs.genieframework.com?packages=Strided
 -->
<!-- [![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/StridedViews.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/report.html) -->

StridedViews.jl exports a single struct type `StridedView` for representing a strided view over a contiguous parrent array, as represented by the abstract type `DenseArray`.

See [Strided.jl](http://github.com/Jutho/Strided.jl) for more functionality.

---

The type `StridedView` provides a view into a parent array of type `DenseArray` such that
the resulting view is strided, i.e. any dimension has an associated stride, such that e.g.
```julia
getindex(A, i₁, i₂, i₃, ...) = A.op(A.parent[offset + 1 + (i₁-1)*s₁ + (i₂-1)*s₂ + (i₃-1)*s₃ + ...])
```
with `sⱼ = stride(A, iⱼ)`. There are no further assumptions on the strides, e.g. they are
not assumed to be monotonously increasing or have `s₁ == 1`. Furthermore, `A.op` can be
any of the operations `identity`, `conj`, `transpose` or `adjoint` (the latter two are
equivalent to the former two if `eltype(A) <: Number`). Since these operations are their own
inverse, they are also used in the corresponding `setindex!`.

This definition enables a `StridedView` to be lazy (i.e. returns just another `StridedView`
over the same parent data) under application of `conj`, `transpose`, `adjoint`,
`permutedims` and indexing (`getindex`) with
`Union{Integer, Colon, AbstractRange{<:Integer}}` (a.k.a slicing). The function `sview` is
exported to directly create a sliced (and thus strided) view over a given parent array.

Furthermore, the strided structure can be retained under certain `reshape` operations, but
not all of them. Any dimension can always be split into smaller dimensions, but two
subsequent dimensions `i` and `i+1` can only be joined if
`stride(A,i+1) == size(A,i)*stride(A,i)`. Instead of overloading `reshape`, Strided.jl
provides a separate function `sreshape` which returns a `StridedView` over the same parent
data, or throws a runtime error if this is impossible.
