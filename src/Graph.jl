using Printf
using Term
using MacroTools

using Unrolled

#################
## Node interface
#################

"""All nodes"""
abstract type AbstractNode end

export nodeValue
"""
    nodeValue(node)

Returns the current value of a node (without any evaluation)
"""
@inline nodeValue(node::Node) where {Node<:AbstractNode} = node.value

"""
    size(node)
    size(graph, node)

Returns a tuple with the size of the array associated with a node of a computation graph.
"""
@inline Base.size(node::Node) where {Node<:AbstractNode} = size(nodeValue(node))

"""
    size(node, dim)
    size(graph, node, dim)

Returns the size of the array associated with a node of a computation graph, along dimension `dim`.
"""
@inline Base.size(node::Node, d::Int) where {Node<:AbstractNode} = size(nodeValue(node), d)

"""
    length(node)
    length(graph, node)
    
Returns the number of entries of the array associated with a node of a computation graph.
"""
@inline Base.length(node::Node) where {Node<:AbstractNode} = Base.prod(size(node))

"""
    similar(node)
    similar(graph, node)

Creates an uninitialized array with the same type and size as the graph node.
"""
@inline Base.similar(node::Node) where {Node<:AbstractNode} = similar(nodeValue(node))

"""
    eltype(node)

Returns the type of the entries of a node.
"""
@inline Base.eltype(node::Node) where {Node<:AbstractNode} = eltype(nodeValue(node))

export typeofvalue
"""Type of the node's value."""
@inline typeofvalue(node::Node) where {Node<:AbstractNode} = typeof(nodeValue(node))

####################
## Computation Graph
####################

export ComputationGraph, resetLog!
"""
Structure use to store the computation graph.

# Fields:
+ `nodes::Vector{AbstractNode}`: vector with all nodes in the graph in topological order: children
        always appear after parents.

+ `children::Vector{Vector{Int}}`: vector with all the children of each node
+ `parents::Vector{Vector{Int}}`: vector with all the parents of each node

+ `validValue::BitVector`: boolean vector, `true` indicates that node contains a valid value
+ `compute_with_ancestors::Vector{FunctionWrapper}: vector of function to compute each node all the
  required ancestors

+ `count::Vector{UInt}`: number of times each node has been computed since last `resetLog!`
"""
struct ComputationGraph{TypeValue}
    nodes::Vector{AbstractNode}
    # dependencies
    children::Vector{Vector{Int}}
    parents::Vector{Vector{Int}}
    validValue::BitVector
    compute_with_ancestors::Vector{FunctionWrapper}
    # array with all the data
    storage::Vector{Any}
    # logging
    count::Vector{UInt}
    time::Vector{UInt64}
    # for parallel computation
    tasks::Vector{Union{Threads.Task,Nothing}}
    enableTask::BitVector
    requestEvent::Vector{Threads.Event}
    validEvent::Vector{Threads.Event}
    ComputationGraph{TypeValue}() where {TypeValue} =
        new{TypeValue}(
            AbstractNode[],
            Vector{UInt}[], Vector{UInt}[],
            BitVector(undef, 0),
            FunctionWrapper[],
            Any[],
            UInt[],
            UInt64[],
            Union{Threads.Task,Nothing}[],
            BitVector(undef, 0),
            Vector{Threads.Event}[],
            Vector{Threads.Event}[],
        )
end

"""Number of nodes in the graph."""
@inline Base.length(graph::ComputationGraph{TypeValue}) where {TypeValue} = length(graph.nodes)

"""Total memory for all the variables stored in the graph."""
@inline memory(graph::ComputationGraph{TypeValue}) where {TypeValue} =
    sum(Base.summarysize(nodeValue(node)) for node in graph.nodes; init=0)


@inline function resetLog!(graph::ComputationGraph{TypeValue}) where {TypeValue}
    graph.time .= zero(UInt64)
    graph.count .= zero(UInt)
end

"""List with all the parents of a set of node."""
function nodesAndParents(
    graph::ComputationGraph{TypeValue},
    nodes::Vararg{AbstractNode}
) where {TypeValue}
    ids = reduce(union, union(Set(node.id), Set(graph.parents[node.id]))
                        for node in nodes; init=Set{Int}()) |> collect |> sort
    return ids
end
export children
"""List with all the children of a set of node."""
function children(
    graph::ComputationGraph{TypeValue},
    nodes::Vararg{AbstractNode}
) where {TypeValue}
    ids = reduce(union, Set(graph.children[node.id])
                        for node in nodes; init=Set{Int}()) |> collect |> sort
    return ids
end

"""
Add node to graph (avoiding repeated nodes).
"""
Base.push!(
    graph::ComputationGraph{TypeValue},
    type::Type,
    computeFunction::Function,
    parameters::TP,
    parentIds::TPI,
    parentValues::TPV,
    dims::NTuple{N,Int}
) where {TypeValue,TPI<:Tuple,TPV<:Tuple,TP<:Tuple,N} =
    push!(graph, type, computeFunction, parameters, parentIds, parentValues, TypeArray{TypeValue,N}(undef, dims), false)
Base.push!(
    graph::ComputationGraph{TypeValue},
    type::Type,
    computeFunction::Function,
    parameters::TP,
    parentIds::TPI,
    parentValues::TPV,
    typeValue::DataType,
    dims::NTuple{N,Int}
) where {TypeValue,TPI<:Tuple,TPV<:Tuple,TP<:Tuple,N} =
    push!(graph, type, computeFunction, parameters, parentIds, parentValues, typeValue{TypeValue,N}(undef, dims), false)
function Base.push!(
    graph::ComputationGraph{TypeValue},
    type::Type,
    computeFunction::Function,
    parameters::TP,
    parentIds::TPI,
    parentValues::TPV,
    value::TV,
    validValue::Bool
) where {TypeValue,TP<:Tuple,TPI<:Tuple,TPV<:Tuple,TV<:AbstractArray}
    # check whether node already exists
    for (i, node) in enumerate(graph.nodes)
        #println(type)
        if isa(node, type) &&                     # match type
           node.parameters == parameters &&       # parameters
           node.parentIds == parentIds &&         # parents
           size(node) == size(value) &&           # size
           # variables need to point to the same data (checked with ===)
           (!(type <: NodeVariable) || nodeValue(node) === value) &&
           # constants must match values (checked with ==)
           (!(type <: NodeConstant) || nodeValue(node) == value)
            #@warn("reusing node ", n)
            return node
        end
    end

    # check special nodes
    for pid in parentIds
        @assert !specialComputation(graph.nodes[pid]) "$(type)($(typeof.(graph.nodes[collect(parentIds)]))) not implemented"
    end

    # next node id
    id = length(graph.nodes) + 1

    # generate computation function
    (compute_with_ancestors, compute_node) = generateComputeFunctions(
        graph, id, computeFunction, parameters, parentIds, parentValues, value)

    # add node to graph
    node = type(id, parameters, parentIds, value, compute_node)
    push!(graph.nodes, node)

    # update children
    push!(graph.children, Int[])
    add2children(graph, node, id)

    # update parents
    directParents = Set(id for id in parentIds if !noComputation(graph.nodes[id]))
    allParents = reduce(union,
                     Set(graph.parents[id])
                     for id in parentIds; init=directParents) |> collect |> sort
    push!(graph.parents, allParents)

    # update valid flags
    push!(graph.validValue, validValue)

    # update computation functions
    push!(graph.compute_with_ancestors, compute_with_ancestors)

    # update storage
    push!(graph.storage, value)

    # update logging
    push!(graph.time, zero(UInt64))
    push!(graph.count, zero(UInt))

    # tasks
    push!(graph.tasks, nothing)
    push!(graph.enableTask, false)
    push!(graph.requestEvent, Threads.Event(true))
    push!(graph.validEvent, Threads.Event(false))

    return node
end

"""
Add node id to all its parents, parents's parents, etc.
"""
function add2children(
    graph::ComputationGraph{TypeValue},
    node::AbstractNode,
    id::Int
) where {TypeValue}
    for pid in node.parentIds
        if !(id in graph.children[pid])
            push!(graph.children[pid], id)
        end
        add2children(graph, graph.nodes[pid], id)
    end
end

export @add
"""
    @add graph expression

Macro to add a complex expression into a computation graph. 

This macro "breaks" the complex expression to elementary subexpressions and add them all to the graph.

# Parameters:
+ `graph::ComputationGraph{TypeValue}`: graph where expression will be stored
+ `expression::Expr`: expression to be added to the graph

# Returns:
+ `Node::AbstractNode`: graph node for the final expression 

# Example:

The following code provides two alternatives to create a computation graph to evaluate
    err = ||A *x -b ||^2

1) without the @add macro
    ```julia
    using ComputationGraphs
    gr=ComputationGraph{Float32}()
    A = variable(gr,3,4)
    x = variable(gr,4)
    b = variable(gr,3)
    Ax  = *(gr,A,x)
    Axb = -(gr,Ax,b)
    err = norm2(gr,Axb)
    display(gr)
    ```
2) without the @add macro
    ```julia
    using ComputationGraphs
    gr=ComputationGraph{Float32}()
    A = @add gr variable(3,4)
    x = @add gr variable(4)
    b = @add gr variable(3)
    err = @add gr norm2(times(A,x)-b)
    display(gr)
    ```

"""
macro add(graph, expression)
    #@show graph
    #@show expression
    newExpression = MacroTools.postwalk(expression) do x
        if @capture(x, xs_')
            # transpose is special case since does not generate :call
            new_x = Expr(:call, adjoint, graph, xs)
            return new_x
            ## deal with associativity rule (?)
        elseif @capture(x, x1_ + x2_)
            new_x = Expr(:call, plus, graph, x1, x2)
            return new_x
        elseif @capture(x, x1_ - x2_)
            new_x = Expr(:call, subtract, graph, x1, x2)
            return new_x
            ## broadcasting
        elseif @capture(x, x1_ .+ x2_)
            new_x = Expr(:call, broadcast, +, graph, x1, x2)
            return new_x
        elseif @capture(x, x1_ .- x2_)
            new_x = Expr(:call, broadcast, -, graph, x1, x2)
            return new_x
        elseif @capture(x, x1_ .* x2_)
            new_x = Expr(:call, broadcast, *, graph, x1, x2)
            return new_x
            ## fall back
        elseif @capture(x, f_(xs__))
            #@show x
            #@show f
            #@show xs
            #@show esc.(xs)
            new_x = Expr(:call, f, graph, xs...)
            #@show new_x
            return new_x
        else
            #@show x
            return x
        end
    end
    #@show newExpression
    return esc(newExpression)
end

# Adding graph-version of node interface

@inline nodeValue(::ComputationGraph{TypeValue}, node::Node
) where {TypeValue,Node<:AbstractNode} = nodeValue(node)
@inline nodeValue(::ComputationGraph{TypeValue}, nodes::Tuple
) where {TypeValue} = Tuple(nodeValue(node) for node in nodes)
@inline nodeValue(::ComputationGraph{TypeValue}, nodes::NamedTuple
) where {TypeValue} = (; (k => nodeValue(nodes[k]) for k in eachindex(nodes))...)

@inline Base.size(graph::ComputationGraph{TypeValue}, node::Node
) where {TypeValue,Node<:AbstractNode} = size(node)
@inline Base.size(graph::ComputationGraph{TypeValue}, node::Node, d::Int
) where {TypeValue,Node<:AbstractNode} = size(node, d)
@inline Base.length(graph::ComputationGraph{TypeValue}, node::Node,
) where {TypeValue,Node<:AbstractNode} = length(node)

@inline Base.similar(::ComputationGraph{TypeValue}, node::Node
) where {TypeValue,Node<:AbstractNode} = similar(node)
@inline Base.eltype(::ComputationGraph{TypeValue}, node::Node
) where {TypeValue,Node<:AbstractNode} = eltype(node)
@inline typeofvalue(::ComputationGraph{TypeValue}, node::Node
) where {TypeValue,Node<:AbstractNode} = typeof(node)

####################
## Display functions
####################

shortTypeof(node::Node) where {Node<:AbstractNode} =
    lowercasefirst(replace(string(typeof(node)), "ComputationGraphs.Node" => ""))

"""
    display(node)
    display(nodes)

Display one node of a computation graph or a tuple of nodes
"""
function Base.display(node::AbstractNode)
    @printf("%3d: %-25s %s size=%s parentIds=[%s]",
        node.id, (@blue shortTypeof(node)), typeofvalue(node), size(node), node.parentIds)
    if length(nodeValue(node)) <= 1
        @printf(" val=%s\n", nodeValue(node))
    elseif length(nodeValue(node)) < 5
        @printf("\n\tval=%s\n", nodeValue(node))
    else
        println()
    end
end
function Base.display(nodes::Tuple)
    println("Tuple{Nodes}=")
    for node in nodes
        display(node)
    end
end
function Base.display(nodes::NamedTuple)
    println("NamedTuple{Nodes}=")
    for key in eachindex(nodes)
        @printf("%-10s: ", key)
        display(nodes[key])
    end
end
function Base.display(nodes::AbstractArray{Node}) where {Node<:AbstractNode}
    println(typeof(nodes), "=")
    for key in eachindex(nodes)
        @printf("%3s: ", key)
        display(nodes[key])
    end
end

"""
    display(graph;topTimes=false)

Display the nodes of a computation graph.

When `topTimes=true` only displays the nodes with the largest total computation times (and hides
information about parents/children).
"""
function Base.display(graph::ComputationGraph{TypeValue};
    topTimes=false, percentage=0.8) where {TypeValue}
    @printf("%s [values use %d bytes/%d bytes total]\n",
        typeof(graph), memory(graph), Base.summarysize(graph))
    if topTimes
        maxTime = maximum(graph.time)
        which = findall(graph.time .>= percentage * Int(maxTime))
        @printf("  %d top computation nodes:\n", length(which))
    else
        which = 1:length(graph.nodes)
    end
    for id in which
        node = graph.nodes[id]
        @printf("  %3d %c: %-25s (%-18s, size=%-10s",
            id, graph.validValue[id] ? '*' : ' ',
            (@blue shortTypeof(node)), typeofvalue(node), size(node))
        if graph.count[id] > 0
            @printf("%8.2fms,%8.2fus/cnt=%7d)\n",
                1e-6graph.time[id], 1e-3graph.time[id] / graph.count[id], graph.count[id])
        else
            println(")")
        end
        if !topTimes
            print("                                           parents = [")
            for pid in node.parentIds
                print(@green shortTypeof(graph.nodes[pid]) * "[" * string(pid) * "],")
            end
            println("]")
            if !isempty(graph.children[id])
                if length(graph.children[id]) < 10
                    @printf("                                           all children =[%s]\n",
                        @magenta join(string.(graph.children[id]), ","))
                else
                    @printf("                                           all children =[%s]\n",
                        @magenta string(length(graph.children[id])) * " nodes")
                end
            end
            if !isempty(graph.parents[id])
                if length(graph.parents[id]) < 10
                    @printf("                                           all parents  =[%s]\n",
                        @blue join(string.(graph.parents[id]), ","))
                else
                    @printf("                                           all parents  =[%s]\n",
                        @blue string(length(graph.parents[id])) * " nodes")
                end
            end
            if length(nodeValue(node)) < 5
                @printf("\t val=%s\n", string(nodeValue(node)))
            end
        end
    end
end

"""
    display(graph,node;withParents=true)

When `withParents=true` shows the full expression needed compute a specific node, otherwise only
shows the specific node (as in `display(node)`).
"""
function Base.display(
    graph::ComputationGraph{TypeValue},
    node::Node;
    withParents=true,
    shown=nothing
) where {TypeValue,Node<:AbstractNode}
    if withParents
        indent = 0
        if isnothing(shown)
            shown = falses(length(graph))
        end
        display_(graph, node, indent, shown)
    else
        display(node)
    end
end
# FIXME should use Vararg
function Base.display(
    graph::ComputationGraph{TypeValue},
    nodes::Vararg{AbstractNode};
    withParents=true,
    shown=nothing
) where {TypeValue}
    if isnothing(shown)
        shown = falses(length(graph))
    end
    for node in nodes
        display(graph, node; withParents, shown)
    end
end

function display_(
    graph::ComputationGraph{TypeValue},
    node::Node,
    indent::Int,
    shown::BitVector;
    noindent=false
) where {TypeValue,Node<:AbstractNode}
    id = node.id
    if noindent
        indentStr = ""
    else
        indentStr = @sprintf("%3d: %s", id, repeat(" ", indent))
    end
    if noComputation(node)
        @printf("%s%s:%d\n", indentStr, (@green shortTypeof(node)), id)
    elseif shown[id]
        @printf("%s%s:%d\n", indentStr, (@magenta shortTypeof(node)), id)
    else
        shown[id] = true
        if length(node.parentIds) == 1
            @printf("%s%s:%d(", indentStr, (@blue shortTypeof(node)), id)
            pid = node.parentIds[1]
            display_(graph, graph.nodes[pid], indent + 4, shown, noindent=true)
        else
            @printf("%s%s:%d(\n", indentStr, (@blue shortTypeof(node)), id)
            for pid in node.parentIds
                display_(graph, graph.nodes[pid], indent + 4, shown)
            end
        end
    end
end
