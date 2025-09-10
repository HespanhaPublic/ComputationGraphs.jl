module ComputationGraphs

const TypeArray = Array

# Specialized version of `FunctionWrappers`
include(raw"FunctionWrappers.jl")

# Graph creation/computation
include(raw"Graph.jl")

include(raw"newnode_macros.jl")

include(raw"Compute.jl")

# Node creation
include(raw"Variables.jl")

include(raw"LinearAlgebra.jl")
include(raw"Functions.jl")

## Node classification for symbolic manipulations
"""Nodes that never change (no sets & zero derivative)"""
AbstractConstantNode = Union{
    NodeConstant,
    NodeZeros,
    NodeOnes,
    NodeUnitVector}
zeroDerivative(node::Node) where {Node<:AbstractNode} = isa(node, AbstractConstantNode)
"""Nodes for which "shortcuts" in computation are possible"""
AbstractSpecialNode = Union{
    NodeZeros,
    NodeOnes,
    NodeUnitVector,
    NodeExpandColumns,
    NodeAdjoint_, # TODO add adjoint ???
}
specialComputation(node::Node) where {Node<:AbstractNode} = isa(node, AbstractSpecialNode)
"""Nodes that do not require re-computation after creation"""
noComputation(node::Node) where {Node<:AbstractNode} = isempty(node.parentIds)

include(raw"SymbolicSimplifications.jl")
include(raw"Differentiation.jl")

include(raw"CodeGeneration.jl")

include(raw"RecipesOptimization.jl")
include(raw"RecipesNN.jl")

include(raw"NN.jl")

end # module ComputationGraphs
