module StridedViewsCUDAExt

using StridedViews
using CUDA
using CUDA: Adapt, CuPtr

const CuStridedView{T, N, A <: CuArray{T}} = StridedView{T, N, A}

function Adapt.adapt_structure(to, A::CuStridedView)
    return StridedView(
        Adapt.adapt_structure(to, parent(A)),
        A.size, A.strides, A.offset, A.op
    )
end

function Base.pointer(x::CuStridedView{T}) where {T}
    return Base.unsafe_convert(CuPtr{T}, pointer(x.parent, x.offset + 1))
end
function Base.unsafe_convert(::Type{CuPtr{T}}, a::CuStridedView{T}) where {T}
    return convert(CuPtr{T}, pointer(a))
end

function Base.print_array(io::IO, X::CuStridedView)
    return Base.print_array(io, Adapt.adapt_structure(Array, X))
end

end # module
