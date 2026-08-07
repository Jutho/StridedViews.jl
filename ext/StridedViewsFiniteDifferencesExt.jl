module StridedViewsFiniteDifferencesExt

using StridedViews: StridedView
using FiniteDifferences

function FiniteDifferences.to_vec(x::StridedView)
    x_vec, from_vec = FiniteDifferences.to_vec(Array(x))
    StridedView_from_vec(x_vec) = StridedView(from_vec(x_vec))
    return x_vec, StridedView_from_vec
end

end
