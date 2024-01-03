module StridedViewsFillArraysExt

using StridedViews
using FillArrays
using FillArrays: AbstractFill, getindex_value

_strides(A::AbstractFill) = (1, Base.size_to_strides(size(A)...)...)
_strides(::AbstractFill{T,0}) where {T} = ()

function StridedViews.StridedView(parent::A, sz::NTuple{N,Int}=size(parent),
                                  st::NTuple{N,Int}=_strides(parent),
                                  offset::Int=0, op::F=identity) where {A<:AbstractFill,N,F}
    T = Base.promote_op(op, eltype(parent))
    return StridedView{T,N,A,F}(parent, sz, st, offset, op)
end

function FillArrays.getindex_value(a::StridedView{T,N,A}) where {T,N,A<:AbstractFill}
    return a.op(getindex_value(parent(a)))
end

# short-circuit indexing to only call checkbounds, no index computation needed
@inline function Base.getindex(a::StridedView{T,N,A},
                               I::Vararg{Int,N}) where {T,N,A<:AbstractFill}
    @boundscheck checkbounds(a, I...)
    return getindex_value(a)
end
@inline function Base.setindex!(a::StridedView{T,N,A}, v,
                                I::Vararg{Int,N}) where {T,N,A<:AbstractFill}
    @boundscheck checkbounds(a, I...)
    v == getindex_value(a) ||
        throw(ArgumentError("Cannot setindex! to $v for an AbstractFill with value $(getindex_value(a))."))
    return a
end

end
