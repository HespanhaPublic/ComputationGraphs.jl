using Revise

include(raw"test_newnode_macros.jl")
include(raw"test_Variables.jl")

include(raw"test_Functions.jl")
include(raw"test_LinearAlgebra.jl")

include(raw"test_Simplifications.jl")
include(raw"test_Differentiation.jl")

include(raw"test_NN.jl")
include(raw"test_NNparallel.jl")

include(raw"test_Hessian.jl")

include(raw"test_CodeGeneration_struct.jl")
include(raw"test_CodeGeneration_module.jl")
