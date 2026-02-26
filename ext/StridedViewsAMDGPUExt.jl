module StridedViewsAMDGPUExt

using StridedViews
using AMDGPU
using AMDGPU: Adapt, ROCPtr

const ROCStridedView{T, N, A <: ROCArray{T}} = StridedView{T, N, A}

function Adapt.adapt_structure(to, A::ROCStridedView)
    return StridedView(
        Adapt.adapt_structure(to, parent(A)),
        A.size, A.strides, A.offset, A.op
    )
end

function Base.pointer(x::ROCStridedView{T}) where {T}
    return Base.unsafe_convert(Ptr{T}, pointer(x.parent, x.offset + 1))
end
function Base.unsafe_convert(::Type{Ptr{T}}, a::ROCStridedView{T}) where {T}
    return convert(Ptr{T}, pointer(a))
end

function Base.print_array(io::IO, X::ROCStridedView)
    return Base.print_array(io, Adapt.adapt_structure(Array, X))
end

end # module
