using Unrolled

#################
## Variable nodes
#################

@inline cg_nothing!(val::TV) where {TV<:AbstractArray} = nothing

@doc """
    variable(graph,dims)
    variable(graph,dims...)
    variable(graph, value)
    
Creates a variable of the given dimension

# Parameters:
+ `graph::ComputationGraph`: Computation graph where variable will be stored.
+ `dims::NTuple{N,Int}`: Desired dimension of the variable. An empty tuple () results in a scalar variable.
+ `value::AbstractArray`: Initial value for the variable, which implicitly defines its dimension.
        If `value` is a scalar, it is first converted to a 0-dimensional array using `fill(value)`.

# Returns:
+ node of the computation graph associated with the variable created
""" ComputationGraphs.variable

export variable
""" 
Node of a computation graph used to represent a variable whose value can
be directly set. It is created by variable().
"""
struct NodeVariable{TP<:Tuple,TPI<:Tuple,TV<:AbstractArray,TC} <: AbstractNode
    id::Int
    parameters::TP
    parentIds::TPI
    value::TV
    compute!::TC
end
variable(graph::ComputationGraph, dims::NTuple{N,Int}) where {N} =
    push!(graph, NodeVariable, cg_variable!, (), (), (), dims)
variable(graph::ComputationGraph, type::DataType, dims::NTuple{N,Int}) where {N} =
    push!(graph, NodeVariable, cg_variable!, (), (), (), type, dims)
function variable(graph::ComputationGraph, val::TV) where {TV<:AbstractArray}
    if eltype(val) != graph.TypeValue
        #error("variable: type of value does not match graph's default type")
        @warn("variable: value type=$(eltype(val)) does not match graph's default type $(graph.TypeValue), you can explicitly include the type in variable() to avoid this warning")
    end
    return push!(graph, NodeVariable, cg_variable!, (), (), (), val, true)
end
function variable(graph::ComputationGraph, type::DataType, val::TV
) where {TV<:AbstractArray}
    if eltype(val) != type
        error("variable: type=$(type) does not match value type=$(eltype(val))")
    end
    return push!(graph, NodeVariable, cg_variable!, (), (), (), val, true)
end
variable(graph::ComputationGraph, val::V) where {V<:Number} =
    variable(graph, fill(val)) # scalar


# shortcuts for scalars, vectors, and matrices
variable(graph::ComputationGraph) = variable(graph, ())
variable(graph::ComputationGraph, n1::Int) = variable(graph, (n1,))
variable(graph::ComputationGraph, n1::Int, n2::Int) = variable(graph, (n1, n2))
variable(graph::ComputationGraph, type::DataType) = variable(graph, type, ())
variable(graph::ComputationGraph, type::DataType, n1::Int) = variable(graph, type, (n1,))
variable(graph::ComputationGraph, type::DataType, n1::Int, n2::Int) =
    variable(graph, type, (n1, n2))

function cg_variable!(val::TV) where {TV<:AbstractArray}
    error("trying to \"compute\" un-initialized variable (type=$(typeof(val)), size=$(size(val))")
    return nothing
end

@inline compute!(node::NodeVariable) =
    error("trying to \"compute\" un-initialized variable (node.id=$(node.id))")

#############
## set values
#############

export set!

"""
    set!(graph,node,value)
    set!(graph,nodes,values)

Update a variable node
+ set value of a variable node 
+ mark all the children as having invalid values
"""
function set!(
    graph::ComputationGraph,
    node::NodeVariable{Tuple{},Tuple{},TV,TC},
    value::TV2 # allow for different (but compatible) abstract arrays
) where {TV<:AbstractArray,TC,TV2<:AbstractArray}
    t0 = time_ns()
    id::Int = node.id
    set_node!(node, value)
    # all children need to be recomputed
    for cid::Int in graph.children[id]
        graph.validValue[cid] = false
    end
    graph.validValue[id] = true
    graph.time[id] += (time_ns() - t0)
    graph.count[id] += 1
    return nothing
end
# allow scalar input
function set!(
    graph::ComputationGraph,
    node::NodeVariable{Tuple{},Tuple{},TV,TC},
    value::V
) where {TV<:AbstractArray,V<:Number,TC}
    t0 = time_ns()
    id::Int = node.id
    set_node!(node, value)
    # all children need to be recomputed
    for cid::Int in graph.children[id]
        graph.validValue[cid] = false
    end
    graph.validValue[id] = true
    graph.time[id] += (time_ns() - t0)
    graph.count[id] += 1
    return nothing
end
@unroll function set!(
    graph::ComputationGraph,
    nodes::NTuple{N,Any},
    values::NTuple{N,Any},
) where {N}
    @unroll for k in 1:length(nodes)
        @inbounds node = nodes[k]
        t0 = time_ns()
        id::Int = node.id
        @inbounds set_node!(node, values[k])
        # all children need to be recomputed
        for cid::Int in graph.children[node.id]
            graph.validValue[cid] = false
        end
        graph.validValue[id] = true
        graph.time[id] += (time_ns() - t0)
        graph.count[id] += 1
    end
    return nothing
end
@unroll function set!(
    graph::ComputationGraph,
    # Names must match (in order to be able to use unroll)
    nodes::NamedTuple{Names,Values1},
    values::NamedTuple{Names,Values2},
) where {Names,Values1,Values2}
    @unroll for k in 1:length(nodes) # only valid if names match in order
        #for k in eachindex(destinations) # this would allow unmatched orders, but @unroll would not work
        @inbounds node = nodes[k]
        t0 = time_ns()
        id::Int = node.id
        @inbounds set_node!(node, values[k])
        # all children need to be recomputed
        for cid::Int in graph.children[node.id]
            graph.validValue[cid] = false
        end
        graph.validValue[node.id] = true
        graph.time[id] += (time_ns() - t0)
        graph.count[id] += 1
    end
    return nothing
end


# not exported since "dangerous" as it does not mark the graph
@inline function set_node!(
    node::NodeVariable{Tuple{},Tuple{},TV,TC},
    value::TV2 # allow for different (but compatible) abstract arrays
) where {TV<:AbstractArray,TC,TV2<:AbstractArray}
    val = nodeValue(node)
    @assert size(val) === size(value) "size mismatch in set!: $(size(val)) != $(size(value))" # TODO must check allocations
    copyto!(val, value)
    return nothing
end
# allow scalar input
@inline function set_node!(
    node::NodeVariable{Tuple{},Tuple{},TV,TC},
    value::V
) where {TV<:AbstractArray,TC,V<:Number}
    @assert size(node) === () "size mismatch in set!: $(size(node)) not compatible with scalar"
    nodeValue(node)[1] = value
    return nothing
end


##############
## copy values
##############

"""
+ performing whatever computations are need for source node to be valid
+ copy value of source to destination node
+ mark all children of the destination node as having invalid values
"""
function Base.copyto!(
    graph::ComputationGraph,
    dest::NodeVariable{Tuple{},Tuple{},TV,TC},
    src::Node
) where {TV<:AbstractArray,TC,Node<:AbstractNode}
    #copyto!(dest.value, src.value)
    t0 = time_ns()
    copyto_node!(dest, src) # TODO faster ???
    id::Int = dest.id
    for cid::Int in graph.children[id]
        graph.validValue[cid] = false
    end
    graph.validValue[id] = true
    graph.time[id] += (time_ns() - t0)
    graph.count[id] += 1
    return nothing
end
# tuple version
@unroll function Base.copyto!(
    graph::ComputationGraph,
    destinations::NTuple{N,Any},
    sources::NTuple{N,Any},
) where {N}
    @unroll for k in 1:length(destinations)
        t0 = time_ns()
        @inbounds copyto_node!(destinations[k], sources[k]) # TODO faster ???
        id::Int = destinations[k].id
        for cid::Int in graph.children[id]
            graph.validValue[cid] = false
        end
        graph.validValue[id] = true
        graph.time[id] += (time_ns() - t0)
        graph.count[id] += 1
    end
    return nothing
end

@unroll function Base.copyto!(
    graph::ComputationGraph,
    # Names must match (in order to be able to use unroll)
    destinations::NamedTuple{Names,Values1},
    sources::NamedTuple{Names,Values2},
) where {Names,Values1,Values2}
    @unroll for k in 1:length(destinations) # only valid if names match in order
        #for k in eachindex(destinations) # this would allow unmatched orders, but @unroll would not work
        t0 = time_ns()
        copyto_node!(destinations[k], sources[k]) # TODO faster ???
        id::Int = destinations[k].id
        for cid::Int in graph.children[id]
            graph.validValue[cid] = false
        end
        graph.validValue[id] = true
        graph.time[id] += (time_ns() - t0)
        graph.count[id] += 1
    end
    return nothing
end

# without graph (not exported since "dangerous" as it does not mark invalidValues)
@inline function copyto_node!(
    dest::NodeVariable{Tuple{},Tuple{},TV,TC},
    src::Node
) where {TV<:AbstractArray,TC,Node<:AbstractNode}
    @assert size(dest) === size(src) "size mismatch in copyto!" # TODO must check allocations
    copyto!(nodeValue(dest)::TV, nodeValue(src)::TV)
    return nothing
end


#################
## Constant nodes
#################

@doc """
    constant(graph, value)
    
Creates a (constant) array equal to the given value. 

# Parameters:
+ `graph::ComputationGraph`: Computation graph where the array will be stored.
+ `value::AbstractArray`: Desired value for the array. 
        If `value` is a scalar, it is first converted to a 0-dimensional array using `fill(value)`.

# Returns:
+ node of the computation graph associated with an array created
""" ComputationGraphs.constant

## constant
export constant
""" 
Node of a computation graph used to represent a constant whose value cannot be changed. It is created by constant().
"""
struct NodeConstant{TP<:Tuple,TPI<:Tuple,TV<:AbstractArray,TC} <: AbstractNode
    id::Int
    parameters::TP
    parentIds::TPI
    value::TV
    compute!::TC
end
function constant(
    graph::ComputationGraph, val::TV
) where {TV<:AbstractArray}
    if eltype(val) != graph.TypeValue
        #error("constant: type of value does not match graph's default type")
        @warn("constant: value type=$(eltype(val)) does not match graph's default type $(graph.TypeValue), you can explicitly include the type in constant() to avoid this warning")
    end
    return push!(graph, NodeConstant, cg_nothing!, (), (), (), val, true)
end
function constant(
    graph::ComputationGraph, type::DataType, val::TV
) where {TV<:AbstractArray}
    if eltype(val) != type
        error("constant: type=$(type) does not match value type=$(eltype(val))")
    end
    return push!(graph, NodeConstant, cg_nothing!, (), (), (), val, true)
end
constant(graph::ComputationGraph, val::V) where {V<:Number} =
    constant(graph, fill(val)) # scalar

@inline compute!(::NodeConstant) = nothing

#############
## Zero nodes
#############

@doc """
    zeros(graph, dims)
    zeros(graph, dims...) 
    
Creates an array filled with 0's

# Parameters:
+ `graph::ComputationGraph`: computation graph where the array will be stored
+ `dims::NTuple{N,Int}`: dimension of the array

# Returns:
+ node of the computation graph associated with an array filled with `zero(TypeValue)`
""" zeros

""" 
Node of a computation graph used to represent a constant equal to an array of zeros. It is created by zeros().
"""
struct NodeZeros{TP<:Tuple,TPI<:Tuple,TV<:AbstractArray,TC} <: AbstractNode
    id::Int
    parameters::TP
    parentIds::TPI
    value::TV      # wasteful, but currently only way to track dimension
    compute!::TC
end
Base.zeros(graph::ComputationGraph, dims::NTuple{N,Int}) where {N} =
    push!(graph, NodeZeros, cg_nothing!, (), (), (), zeros(graph.TypeValue, dims), true)
Base.zeros(graph::ComputationGraph, type::DataType, dims::NTuple{N,Int}) where {N} =
    push!(graph, NodeZeros, cg_nothing!, (), (), (), zeros(type, dims), true)
# shortcuts for scalars, vectors, and matrices
Base.zeros(graph::ComputationGraph) = zeros(graph, ())
Base.zeros(graph::ComputationGraph, n1::Int) = zeros(graph, (n1,))
Base.zeros(graph::ComputationGraph, n1::Int, n2::Int) = zeros(graph, (n1, n2))
Base.zeros(graph::ComputationGraph, type::DataType) = zeros(graph, type, ())
Base.zeros(graph::ComputationGraph, type::DataType, n1::Int) = zeros(graph, type, (n1,))
Base.zeros(graph::ComputationGraph, type::DataType, n1::Int, n2::Int) =
    zeros(graph, type, (n1, n2))

@inline compute!(::NodeZeros) = nothing

#############
## Ones nodes
#############

@doc """
    ones(graph, dims)
    ones(graph, dims...) 
    
Creates an array filled with 1's

# Parameters:
+ `graph::ComputationGraph`: computation graph where array will be stored
+ `dims::NTuple{N,Int}`: dimension of the array

# Returns:
+ node of the computation graph associated with an array filled with `one(TypeValue)`
""" ones

""" 
Node of a computation graph used to represent a constant equal to an array of ones. It is created by ones().
"""
struct NodeOnes{TP<:Tuple,TPI<:Tuple,TV<:AbstractArray,TC} <: AbstractNode
    id::Int
    parameters::TP
    parentIds::TPI
    value::TV      # wasteful, but currently only way to track dimension
    compute!::TC
end

# TODO These vector should be shortcutted for products
Base.ones(graph::ComputationGraph, dims::NTuple{N,Int}) where {N} =
    push!(graph, NodeOnes, cg_nothing!, (), (), (), ones(graph.TypeValue, dims), true)
Base.ones(graph::ComputationGraph, type::DataType, dims::NTuple{N,Int}) where {N} =
    push!(graph, NodeOnes, cg_nothing!, (), (), (), ones(type, dims), true)
# shortcuts for scalars, vectors, and matrices
Base.ones(graph::ComputationGraph) = ones(graph, ())
Base.ones(graph::ComputationGraph, n1::Int) = ones(graph, (n1,))
Base.ones(graph::ComputationGraph, n1::Int, n2::Int) = ones(graph, (n1, n2))
Base.ones(graph::ComputationGraph, type::DataType) = ones(graph, type, ())
Base.ones(graph::ComputationGraph, type::DataType, n1::Int) = ones(graph, type, (n1,))
Base.ones(graph::ComputationGraph, type::DataType, n1::Int, n2::Int) =
    ones(graph, type, (n1, n2))

@inline compute!(::NodeOnes) = nothing

########################
## Canonical Basis nodes
########################

# TODO These vector are very sparse so there should be shortcuts for most operations
@doc """
    unitvector(dims,k) 
    
Creates the k-th vector of the canonical basis for the linear space with dimension dims.
""" ComputationGraphs.unitvector
export unitvector

"""
Node of a computation graph used to represent vectors of the canonical basis.
"""
struct NodeUnitVector{TP<:Tuple,TPI<:Tuple,TV<:AbstractArray,TC} <: AbstractNode
    id::Int
    parameters::TP
    parentIds::TPI
    value::TV     # wasteful, but currently only way to track dimension
    compute!::TC
end
unitvector(graph::ComputationGraph, dims::NTuple{N,Int}, k::Int) where {N} =
    push!(graph, NodeUnitVector, cg_nothing!, (k,), (), (), unitvector(graph.TypeValue, dims, k), true)
unitvector(graph::ComputationGraph, type::DataType, dims::NTuple{N,Int}, k::Int) where {N} =
    push!(graph, NodeUnitVector, cg_nothing!, (k,), (), (), unitvector(type, dims, k), true)

@inline compute!(::NodeUnitVector) = nothing

function unitvector(TypeValue, dims::NTuple{N,Int}, k::Int) where {N}
    u = zeros(TypeValue, dims)
    u[k] = one(TypeValue)
    return u
end
