module StridedViewsCUDAExt

using StridedViews
using CUDA
using CUDA: Adapt

const CuStridedView{T,N,A<:CuArray{T}} = StridedView{T,N,A}

function Adapt.adapt_structure(::Type{T}, A::StridedView) where {T}
    return StridedView(Adapt.adapt_structure(T, parent(A)), A.size, A.strides, A.offset,
                       A.op)
end

function Base.unsafe_convert(::Type{CUDA.CuPtr{T}}, a::CuStridedView{T}) where {T}
    return pointer(a.parent, a.offset + 1)
end

function Base.print_array(io::IO, X::CuStridedView)
    return Base.print_array(io, Adapt.adapt_structure(Array, X))
end

end # module
