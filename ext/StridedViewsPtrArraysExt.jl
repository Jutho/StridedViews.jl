module StridedViewsPtrArraysExt

using StridedViews
using PtrArrays

StridedViews._normalizeparent(A::PtrArray) = PtrArray(A.ptr, length(A))

end
