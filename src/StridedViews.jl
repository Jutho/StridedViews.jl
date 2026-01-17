module StridedViews

import Base: parent, size, strides, tail, setindex
using Base: @propagate_inbounds, RangeIndex, Dims
const SliceIndex = Union{RangeIndex,Colon}

using LinearAlgebra
export StridedView, sreshape, sview, isstrided

include("auxiliary.jl")
include("stridedview.jl")

using PackageExtensionCompat
function __init__()
    return @require_extensions
end

end
