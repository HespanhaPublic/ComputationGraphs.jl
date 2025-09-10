##################
## Unary operators
##################

# identity()
Base.identity(graph::ComputationGraph, node::Node) where {Node<:AbstractNode} = node

# (zeros(m,n))' = zeros(n,m) 
Base.adjoint(graph::ComputationGraph, a::NodeZeros) =
    zeros(graph, (size(a, 2), size(a, 1)))

# ()''=()
Base.adjoint(graph::ComputationGraph, node::NodeAdjoint_) =
    graph.nodes[node.parentIds[1]]

#######################
## Addition/Subtraction
#######################

# a+0=a
plus(::ComputationGraph, a::Node, ::NodeZeros) where {Node<:AbstractNode} = a
# 0+b=b
plus(::ComputationGraph, ::NodeZeros, b::Node) where {Node<:AbstractNode} = b
# 0+0=0
plus(::ComputationGraph, a::NodeZeros, ::NodeZeros) = a

# +a = a
Base.:+(::ComputationGraph, a::Node) where {Node<:AbstractNode} = a

# a-0=a
subtract(::ComputationGraph, a::Node, ::NodeZeros) where {Node<:AbstractNode} = a
# 0-b= -b 
subtract(graph::ComputationGraph, ::NodeZeros, b::Node) where {Node<:AbstractNode} =
    minus(graph, b)
# 0-0=0
subtract(::ComputationGraph, a::NodeZeros, ::NodeZeros) = a

#######################
## Addition/Subtraction
#######################

# 0(scalar) * B = 0(size(B)) 
scalarTimes(graph::ComputationGraph, ::NodeZeros, b::Node) where {Node<:AbstractNode} =
    zeros(graph, size(b))
# a(scalar) * 0 = 0
scalarTimes(::ComputationGraph, ::Node, b::NodeZeros) where {Node<:AbstractNode} = b
# 0(scalar) * 0 = 0
scalarTimes(::ComputationGraph, ::NodeZeros, b::NodeZeros) = b

# 0 .* B =0
pointTimes(::ComputationGraph, ::Node, b::NodeZeros) where {Node<:AbstractNode} = b
# A .* 0 =0
pointTimes(::ComputationGraph, a::NodeZeros, ::Node) where {Node<:AbstractNode} = a
# 0 .* 0 =0
pointTimes(::ComputationGraph, a::NodeZeros, ::NodeZeros) = a

# A * x = 0
times(graph::ComputationGraph, A::Node, x::NodeZeros) where {Node<:AbstractNode} =
    zeros(graph, (length(size(x)) == 1 ? (size(A, 1),) : (size(A, 1), size(x, 2))))
# 0 * x = 0
times(graph::ComputationGraph, A::NodeZeros, x::Node) where {Node<:AbstractNode} =
    zeros(graph, (length(size(x)) == 1 ? (size(A, 1),) : (size(A, 1), size(x, 2))))
# 0 * 0 = 0
times(graph::ComputationGraph, A::NodeZeros, x::NodeZeros) =
    zeros(graph, (length(size(x)) == 1 ? (size(A, 1),) : (size(A, 1), size(x, 2))))

# A' * 0 = 0
adjointTimes(graph::ComputationGraph, A::Node, x::NodeZeros) where {Node<:AbstractNode} =
    zeros(graph, (length(size(x)) == 1 ? (size(A, 2),) : (size(A, 2), size(x, 2))))
# 0' * x = 0
adjointTimes(graph::ComputationGraph, A::NodeZeros, x::Node) where {Node<:AbstractNode} =
    zeros(graph, (length(size(x)) == 1 ? (size(A, 2),) : (size(A, 2), size(x, 2))))
# 0' * 0 = 0
adjointTimes(graph::ComputationGraph, A::NodeZeros, x::NodeZeros) =
    zeros(graph, (length(size(x)) == 1 ? (size(A, 2),) : (size(A, 2), size(x, 2))))

# x * 0' =0
timesAdjoint(graph::ComputationGraph, x::Node, y::NodeZeros) where {Node<:AbstractNode} = zeros(graph, (size(x, 1), size(y, 1)))
# 0 * y' =0
timesAdjoint(graph::ComputationGraph, x::NodeZeros, y::Node) where {Node<:AbstractNode} = zeros(graph, (size(x, 1), size(y, 1)))
# 0 * 0' =0
timesAdjoint(graph::ComputationGraph, x::NodeZeros, y::NodeZeros) =
    zeros(graph, (size(x, 1), size(y, 1)))

# dot(x,0)=0
dot_(graph::ComputationGraph, x::Node, y::NodeZeros) where {Node<:AbstractNode} = zeros(graph, ())
# dot(0,x)=0
dot_(graph::ComputationGraph, x::NodeZeros, y::Node) where {Node<:AbstractNode} = zeros(graph, ())
# dot(0,0)=0
dot_(graph::ComputationGraph, x::NodeZeros, y::NodeZeros) = zeros(graph, ())

# A * unitvector
times(graph::ComputationGraph, A::Node, u::NodeUnitVector) where {Node<:AbstractNode} =
    column(graph, A, u.parameters[1])

# A * ones
times(graph::ComputationGraph, A::Node, ::NodeOnes) where {Node<:AbstractNode} =
    sumColumns(graph, A)
# unitvector * y'
timesAdjoint(graph::ComputationGraph, u::NodeUnitVector, y::Node) where {Node<:AbstractNode} =
    unitTimesAdjoint(graph, y, size(u, 1), u.parameters[1])

# x * ones'
timesAdjoint(graph::ComputationGraph, x::Node, o::NodeOnes) where {Node<:AbstractNode} =
    timesAdjointOnes(graph, x, length(o))

# A' * expandColumns(x,rows) 
adjointTimes(graph::ComputationGraph, A::Node, e::NodeExpandColumns) where {Node<:AbstractNode} =
    adjointTimesExpandColumns(graph,
        A, graph.nodes[e.parentIds[1]], graph.nodes[e.parentIds[2]])

# expandColumns(x,rows,nRows) * A'
timesAdjoint(graph::ComputationGraph, e::NodeExpandColumns, A::Node) where {Node<:AbstractNode} =
    expandColumnsTimesAdjoint(graph,
        graph.nodes[e.parentIds[1]], A, graph.nodes[e.parentIds[2]], e.parameters[1])

# sumColumns(expandColumns(x,rows))
sumColumns(graph::ComputationGraph, x::NodeExpandColumns) =
    sumExpandColumns(graph,
        graph.nodes[x.parentIds[1]], graph.nodes[x.parentIds[2]], x.parameters[1])
