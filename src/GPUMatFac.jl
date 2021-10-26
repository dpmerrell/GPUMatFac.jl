
module GPUMatFac


include("model.jl")
include("losses.jl")
include("regularizer.jl")
include("fit.jl")
include("impute.jl")
include("transform.jl")
include("model_io.jl")

BLAS.set_num_threads(1)


end # module
