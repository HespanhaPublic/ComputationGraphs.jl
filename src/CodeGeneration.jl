export Code
"""
Structure used to generate dedicated code:

# Fields
- `parallel::Bool=false`: 

        1) When `true` the `valid` flags are implemented with `Threads.Event`s,
            otherwise just a `Bool` 
        
        2) When `true` each node has a `Threads.Task` that computes the node
           as needed. 
        
- `unrolled::Bool=false`: 

        1) When `true`, the code generated for `get` uses a single function with nested `if`    
           statements to compute nodes on demand. 
  
           This can lead to very large functions for big graphs. 
        
           Parallel computation is not supported in this mode.

        2) When `false`, each node has its own `compute` function that (recursively) calls the    
           parents' `compute` functions.

- `count::Bool=true`: When `true`, the generated code includes counters for how many times each
        node's computation function has been called.
"""
mutable struct Code
    type::Symbol # :module or :struct
    graph::ComputationGraph
    name::String
    inits::String
    sets::String
    gets::String
    copies::String
    computes::String
    expressions::String
    unrolled::Bool
    parallel::Bool
    count::Bool
    function Code(graph::ComputationGraph, name::String;
        type::Symbol=:struct,
        unrolled::Bool=false,
        parallel::Bool=false,
        count::Bool=true
    )
        code = new(
            type, graph, name, "", "", "", "", "", "", unrolled, parallel, count)
        return code
    end
end
Base.display(s::Code) = @printf("# Init\n%s# Sets\n%s# Gets\n%s",
    s.inits, s.sets, s.gets)

# Variable name for values from node ID
nodeDef(code::Code, id::Int) = "Node" * string(id)
nodeValue(code::Code, id::Int) = code.name * "." * nodeDef(code, id)
nodeTaskDef(code::Code, id::Int) = "task_" * nodeDef(code, id)
nodeTask(code::Code, id::Int) = code.name * "." * nodeTaskDef(code, id)
# Variable name for valid from node ID
nodeValidDef(code::Code, id::Int) = "Node" * string(id) * "valid"
nodeValid(code::Code, id::Int) = code.name * "." * nodeValidDef(code, id)
# Variable name for compute function from node ID
nodeCompute(code::Code, id::Int) = "compute_" * nodeDef(code, id)
# Array with counts
countsDef(code::Code) = "cg_counts"
counts(code::Code, id::Int) = code.name * "." * @sprintf("cg_counts[%d]", id)

# Write code
export save
function save(filename::String, code::Code)
    open(filename, "w") do io
        if code.type == :module
            println(io, "module " * code.name)
            println(io, "__revise_mode__=:eval # Revise re-evaluates everything")
            println(io, "using ComputationGraphs")
            println(io, "# Init")
            code.inits = nodes_str(code; indent=0)
            print(io, code.inits)
        elseif code.type == :struct
            println(io, "using ComputationGraphs")
            println(io, "@kwdef mutable struct " * uppercasefirst(code.name))
            println(io, "    # Init")
            code.inits = nodes_str(code; indent=4)
            print(io, code.inits)
            println(io, "end")
        else
            error("Unknown code type $(code.type)")
        end
        if !code.unrolled
            println(io, "# Computed")
            str = ""
            for node in code.graph.nodes
                if !noComputation(node) || isa(node, NodeVariable)
                    str *= compute_str_recursive(code, node, indent=0)
                end
            end
            print(io, str)
        end
        if code.parallel
            println(io, "# Parallel computation")
            print(io, compute_str_parallel(code; indent=0))
        end
        println(io, "# Sets")
        print(io, code.sets)
        println(io, "# Gets")
        print(io, code.gets)
        println(io, "# Copies")
        print(io, code.copies)
        println(io, "# Computes")
        print(io, code.computes)
        println(io, "# Expressions")
        print(io, code.expressions)
        if code.type == :module
            println(io, "end")
        elseif code.type == :struct
        else
            error("Unknown code type $(code.type)")
        end
    end
end

#################
## Initialization
#################
"""
Create initialization code
"""
function nodes_str(
    code::Code;
    indent::Int=0
)
    # create node variables
    graph = code.graph
    indent_str = repeat(" ", indent)
    str = ""
    # counts
    str *= indent_str * @sprintf("%s::Vector{Int}=fill(0,%d)\n",
        countsDef(code), length(graph.nodes))
    # values
    for (i, node) in enumerate(graph.nodes)
        sz = size(node)
        if isa(node, NodeZeros)
            str *= indent_str * @sprintf("%s::%s=fill(zero(%s),%s)\n", nodeDef(code, i),
                typeofvalue(node), eltype(node), sz)
        elseif isa(node, NodeOnes)
            str *= indent_str * @sprintf("%s::%s=fill(one(%s),%s)\n", nodeDef(code, i),
                typeofvalue(node), eltype(node), sz)
        elseif isa(node, NodeUnitVector)
            str *= indent_str * @sprintf("%s::%s=unitvector(%s,%s,%d)\n", nodeDef(code, i),
                typeofvalue(node), eltype(node), sz, node.parameters[1])
        elseif isa(node, NodeConstant)
            str *= indent_str * @sprintf("%s::%s=%s\n", nodeDef(code, i),
                typeofvalue(node), nodeValue(node))
        elseif noComputation(node) && !isa(node, NodeVariable)
            error("initialization of $(typeof(node)) not implemented")
        else
            str *= indent_str * @sprintf("%s::%s=%s{%s,%d}(undef,%s)\n", nodeDef(code, i),
                typeofvalue(node), graph.TypeArray, eltype(node), length(sz), sz)
        end
    end
    # valid
    for (id, node) in enumerate(graph.nodes)
        if !noComputation(node) || isa(node, NodeVariable)
            if code.parallel
                # only variables need valid flag to check if they have been set
                str *= indent_str * @sprintf("%s::Threads.Event=Threads.Event(false)\n",
                    nodeValidDef(code, id))
                if !noComputation(node)
                    # autoreset=true => needed only triggers recomputation once
                    str *= indent_str * @sprintf("%s_needed::Threads.Event=Threads.Event(true)\n",
                        nodeValidDef(code, id))
                    str *= indent_str * @sprintf("%s::Threads.Task=Threads.@spawn nothing\n",
                        nodeTaskDef(code, id))
                end
            else
                str *= indent_str * @sprintf("%s::Bool=false\n", nodeValidDef(code, id))
            end
        end
    end
    return str
end

###################
## Compute one node
###################

"""Create string to call function that does the computation"""
function call_gs(
    code::Code,
    node::Node,
) where {Node<:AbstractNode}
    id = node.id
    for pid in node.parentIds
        p = code.graph.nodes[pid]
        if specialComputation(p)
            @error("$(typeof(node))($(typeof.(code.graph.nodes[collect(node.parentIds)]))) not optimally implemented")
        end
    end
    fun = lowercasefirst(replace(string(typeof(node).name.name), "Node" => ""))
    parents = [nodeValue(code, pid) for pid in node.parentIds]
    parameters = [string(p) for p in node.parameters]
    everything = join(vcat(parents, parameters), ",")
    str::String = @sprintf("ComputationGraphs.cg_%s!(%s,%s)",
        fun, nodeValue(code, id), everything)
    if code.count
        str *= @sprintf(";%s += 1\n", counts(code, id))
    else
        str *= "\n"
    end
    return str
end

"""
Create code to compute nodes (for gets)
[single function with nested if's]
"""
function compute_str_unrolled(
    code::Code,
    node::Node,
    indent::Int=4
) where {Node<:AbstractNode}
    if noComputation(node) && !isa(node, NodeVariable)
        error("$(typeof(node)) does not need computation")
    end
    id = node.id
    indent_str = repeat(" ", indent)
    if code.parallel
        str = indent_str * @sprintf("if !%s.set\n", nodeValid(code, id))
    else
        str = indent_str * @sprintf("if !%s\n", nodeValid(code, id))
    end
    for pid in node.parentIds
        p = code.graph.nodes[pid]
        if !noComputation(p) || isa(p, NodeVariable)
            str *= compute_str_unrolled(code, p, indent + 4)
        end
    end
    str *= indent_str * "    " * call_gs(code, node)
    if code.parallel
        str *= indent_str * @sprintf("    notify(%s)\n", nodeValid(code, id))
    else
        str *= indent_str * @sprintf("    %s=true\n", nodeValid(code, id))
    end
    str *= indent_str * "end\n"
    return str
end

"""
Create code to compute nodes (for gets)
[recursive functions]
"""
function compute_str_recursive(
    code::Code,
    node::Node;
    indent::Int=4
) where {Node<:AbstractNode}
    if noComputation(node) && !isa(node, NodeVariable)
        error("$(typeof(node)) does not need computation")
    end
    id = node.id
    indent_str = repeat(" ", indent)
    firstArg = code.type == :struct ? @sprintf("%s::%s,", code.name, uppercasefirst(code.name)) : ""
    str = indent_str * @sprintf("function %s(%s)\n", nodeCompute(code, id), firstArg)
    for pid in node.parentIds
        p = code.graph.nodes[pid]
        if specialComputation(p)
            @error("$(typeof(node))($(typeof.(code.graph.nodes[collect(node.parentIds)]))) not optimally implemented")
        end
        if !noComputation(p) || isa(p, NodeVariable)
            if code.parallel
                str *= indent_str * @sprintf("    %s.set || %s(%s)\n",
                    nodeValid(code, pid), nodeCompute(code, pid), firstArg)
            else
                str *= indent_str * @sprintf("    %s || %s(%s)\n",
                    nodeValid(code, pid), nodeCompute(code, pid), firstArg)
            end
        end
    end
    str *= indent_str * "    " * call_gs(code, node)
    if code.parallel
        str *= indent_str * @sprintf("    notify(%s)\n", nodeValid(code, id))
    else
        str *= indent_str * @sprintf("    %s=true\n", nodeValid(code, id))
    end
    str *= indent_str * "    return nothing\n"
    str *= indent_str * "end\n"
    return str
end

"""
Create parallel code to recompute all nodes
"""
function compute_str_parallel(
    code::Code;
    indent::Int=4
)
    indent_str = repeat(" ", indent)
    firstArg = code.type == :struct ? @sprintf("%s::%s,", code.name, uppercasefirst(code.name)) : ""
    str = ""
    if true
        ## compute all in parallel
        # TODO spawning new threads leads to allocations & slow performance
        str *= indent_str * @sprintf("function computeAll_parallel(%s)\n", firstArg)
        str *= indent_str * "    @sync begin\n"
        for node in code.graph.nodes
            if !noComputation(node) || isa(node, NodeVariable)
                id = node.id
                str *= indent_str * @sprintf("        if !%s.set\n", nodeValid(code, id))
                if isa(node, NodeVariable)
                    str *= indent_str * @sprintf("            error(\"un-initialized %s cannot be computed, missing call to set\")\n",
                        nodeValue(code, id))
                else
                    str *= indent_str * "            Threads.@spawn begin\n"
                    for pid in node.parentIds
                        p = code.graph.nodes[pid]
                        if !noComputation(p)
                            #str *= indent_str * @sprintf("                println(\"wait(%s)\")\n",nodeValid(code, pid))
                            str *= indent_str * @sprintf("                wait(%s)\n",
                                nodeValid(code, pid))
                            #str *= indent_str * @sprintf("                println(\"waited(%s)\")\n",nodeValid(code, pid))
                        end
                    end
                    str *= indent_str * "                " * call_gs(code, node)
                    str *= indent_str * @sprintf("                notify(%s)\n", nodeValid(code, id))
                    #str *= indent_str * @sprintf("                println(\"notified(%s)\")\n",nodeValid(code, id))
                    str *= indent_str * "            end\n"
                end
                str *= indent_str * "        end\n"
            end
        end
        str *= indent_str * "    end\n"
        str *= indent_str * "end # function computeAll_parallel\n"
    end
    if true
        ## launch asynchronous computation of all nodes
        str *= indent_str * @sprintf("function computeAsync_parallel(%s)\n", firstArg)
        for node in code.graph.nodes
            id = node.id
            if isa(node, NodeVariable)
                str *= indent_str * @sprintf("    if !%s.set\n", nodeValid(code, id))
                str *= indent_str * @sprintf("        error(\"un-initialized %s cannot be computed, missing call to set\")\n",
                    nodeValue(code, id))
                str *= indent_str * @sprintf("    end\n")
            elseif !noComputation(node)
                str *= indent_str * @sprintf("    if istaskdone(%s)\n",
                    nodeTask(code, id))
                str *= indent_str * @sprintf("        %s=Threads.@spawn while true\n",
                    nodeTask(code, id))
                str *= indent_str * @sprintf("            wait(%s_needed)\n",
                    nodeValid(code, id))
                str *= indent_str * @sprintf("            if !%s.set\n", nodeValid(code, id))
                for pid in node.parentIds
                    p = code.graph.nodes[pid]
                    if !noComputation(p)
                        str *= indent_str * @sprintf("                if !%s.set\n",
                            nodeValid(code, pid))
                        str *= indent_str * @sprintf("                    notify(%s_needed)\n",
                            nodeValid(code, pid))
                        str *= indent_str * @sprintf("                    wait(%s)\n",
                            nodeValid(code, pid))
                        str *= indent_str * @sprintf("                end\n")
                    end
                end
                str *= indent_str * "                " * call_gs(code, node)
                str *= indent_str * @sprintf("                notify(%s)\n",
                    nodeValid(code, id))
                str *= indent_str * @sprintf("            end\n")
                str *= indent_str * @sprintf("        end # while true\n")
                str *= indent_str * @sprintf("    end # if istaskdone()\n")
            end
        end
        str *= indent_str * "end #function computeAsync_parallel\n"
    end

    return str
end
################################
## set's/get's/copies'/compute's
################################

export sets!
"""
Add set!'s to code
"""
function sets!(
    code::Code,
    args::Vararg{Pair};
    indent::Int=0,
)
    indent_str = repeat(" ", indent)
    firstArg = code.type == :struct ? @sprintf("%s::%s,", code.name, uppercasefirst(code.name)) : ""
    str = ""
    getType(node) = replace(string(typeofvalue(node)),
        "Matrix" => "AbstractMatrix", "Vector" => "AbstractVector")
    for (nodes, name) in args
        @assert isa(name, String) "function name must be a string"
        if isa(nodes, AbstractNode)
            nodes = (nodes,)
        end
        if isa(nodes, Tuple) || isa(nodes, NamedTuple) || isa(nodes, Vector)
            # function declaration (multiple input parameters)
            str *= indent_str * @sprintf("function %s(%s%s)\n",
                name, firstArg,
                join([@sprintf("v%d::%s", k, getType(node))
                      for (k, node) in enumerate(nodes)], ","))
            toInvalidate = Set{Int}()
            for (k, node) in enumerate(nodes)
                str *= indent_str * @sprintf("    copyto!(%s,v%d)\n", nodeValue(code, node.id), k)
                union!(toInvalidate, code.graph.children[node.id])
            end
            # all children need to be recomputed
            for cid in sort(collect(toInvalidate))
                if code.parallel
                    str *= indent_str * @sprintf("    reset(%s)\n", nodeValid(code, cid))
                else
                    str *= indent_str * @sprintf("    %s=false\n", nodeValid(code, cid))
                end
            end
            for node in nodes
                if code.parallel
                    str *= indent_str * @sprintf("    notify(%s)\n", nodeValid(code, node.id))
                else
                    str *= indent_str * @sprintf("    %s=true\n", nodeValid(code, node.id))
                end
            end
        else
            error("illegal use of sets!(typeof(nodes)=$(typeof(nodes)))")
        end
        str *= indent_str * "    return nothing\n"
        str *= indent_str * "end\n"
    end
    code.sets *= str
end

## Gets(nodes) (& compute(nodes)

export gets!
"""
Add get!'s to code
"""
function gets!(
    code::Code,
    args::Vararg{Pair};
    indent::Int=0,
)
    indent_str = repeat(" ", indent)
    firstArg = code.type == :struct ? @sprintf("%s::%s,", code.name, uppercasefirst(code.name)) : ""
    str = ""
    for (nodes, name) in args
        @assert isa(name, String) "function name must be a string"
        # function declaration
        str *= indent_str * @sprintf("@inline function %s(%s)\n", name, firstArg)
        if isa(nodes, AbstractNode)
            nodes = (nodes,)
        end
        if isa(nodes, Tuple) || isa(nodes, NamedTuple) || isa(nodes, Vector)
            # computes
            for node in nodes
                if !noComputation(node) || isa(node, NodeVariable)
                    if code.unrolled
                        str *= compute_str_unrolled(code, node, indent + 4)
                    else
                        if code.parallel
                            str *= @sprintf("    %s.set || %s(%s)\n",
                                nodeValid(code, node.id), nodeCompute(code, node.id), firstArg)
                        else
                            str *= @sprintf("    %s || %s(%s)\n",
                                nodeValid(code, node.id), nodeCompute(code, node.id), firstArg)
                        end
                    end
                end
            end
            # return command
            str *= indent_str * @sprintf("    return (%s)\n",
                join([nodeValue(code, node.id) for node in nodes], ","))
        else
            error("illegal use of gets!(typeof(nodes)=$(typeof(nodes)))")
        end
        str *= indent_str * "end\n"
        # parallel asynchronous version
        if code.parallel
            str *= indent_str * @sprintf("@inline function %sAsync_parallel(%s)\n", name, firstArg)
            if isa(nodes, AbstractNode)
                nodes = (nodes,)
            end
            if isa(nodes, Tuple) || isa(nodes, NamedTuple) || isa(nodes, Vector)
                for node in nodes
                    if !noComputation(node)
                        str *= @sprintf("    notify(%s_needed)\n", nodeValid(code, node.id))
                        #str *= @sprintf("    %s.set || notify(%s_needed)\n", nodeValid(code, node.id), nodeValid(code, node.id))

                    end
                end
                for node in nodes
                    if !noComputation(node)
                        str *= @sprintf("    wait(%s)\n", nodeValid(code, node.id))
                    end
                end
                # return command
                str *= indent_str * @sprintf("    return (%s)\n",
                    join([nodeValue(code, node.id) for node in nodes], ","))
            else
                error("illegal use of gets!(typeof(nodes)=$(typeof(nodes)))")
            end
            str *= indent_str * "end\n"
        end
    end
    code.gets *= str
end

export copies!
"""
Add copyto!'s to code
"""
function copies!(
    code::Code,
    args::Vararg{Pair};
    indent::Int=0,
)
    indent_str = repeat(" ", indent)
    firstArg = code.type == :struct ? @sprintf("%s::%s,", code.name, uppercasefirst(code.name)) : ""
    str = ""
    for (nodes, name) in args
        @assert isa(name, String) "function name must be a string"
        # function declaration
        str *= indent_str * @sprintf("function %s(%s)\n", name, firstArg)
        (sources, destinations) = nodes
        if isa(sources, AbstractNode) && isa(destinations, AbstractNode)
            # copyto! command
            str *= indent_str * @sprintf("    copyto!(%s,%s)\n",
                nodeValue(code, destinations.id), nodeValue(code, sources.id))
            # all children need to be recomputed
            for c in code.graph.children[destinations.id]
                if code.parallel
                    str *= indent_str * @sprintf("    reset(%s)\n", nodeValid(code, c))
                else
                    str *= indent_str * @sprintf("    %s=false\n", nodeValid(code, c))
                end
            end
        elseif (isa(sources, Tuple) || isa(sources, Vector)) &&
               (isa(destinations, Tuple) || isa(destinations, Vector))
            for k in eachindex(sources, destinations)
                # copyto! command
                str *= indent_str * @sprintf("    copyto!(%s,%s)\n",
                    nodeValue(code, destinations[k].id), nodeValue(code, sources[k].id))
            end
            # all children need to be recomputed
            allChildren = reduce(union, Set(code.graph.children[dest.id])
                                        for dest in destinations; init=Set{Int}()) |> collect |> sort
            for cid in allChildren
                if code.parallel
                    str *= indent_str * @sprintf("    reset(%s)\n", nodeValid(code, cid))
                else
                    str *= indent_str * @sprintf("    %s=false\n", nodeValid(code, cid))
                end
            end
            for node in destinations
                if code.parallel
                    str *= indent_str * @sprintf("    notify(%s)\n", nodeValid(code, node.id))
                else
                    str *= indent_str * @sprintf("    %s=true\n", nodeValid(code, node.id))
                end
            end
        else
            error("illegal use of copies!(typeof(sources)=$(typeof(sources)), typeof(destinations)=$(typeof(destinations)))")
        end
        str *= indent_str * "    return nothing\n"
        str *= indent_str * "end\n"
    end
    code.copies *= str
end

export computes!
"""
Add computes's to code
"""
function computes!(
    code::Code,
    args::Vararg{Pair};
    indent::Int=0,
)
    indent_str = repeat(" ", indent)
    firstArg = code.type == :struct ? @sprintf("%s::%s,", code.name, uppercasefirst(code.name)) : ""
    str = ""
    for (nodeIDs, name) in args
        @assert isa(name, String) "function name must be a string"
        # function declaration
        str *= indent_str * @sprintf("function %s(%s)\n", name, firstArg)
        if isa(nodeIDs, AbstractNode)
            nodeIDs = [nodeIDs.id]
            #println(@bold @blue "Node -> $(nodeIDs)")
        elseif !isa(nodeIDs, AbstractVector{Int})
            error("illegal use of computes!!(typeof(nodes)=$(typeof(nodeIDs)))")
        end
        for id in sort(nodeIDs)
            node = code.graph.nodes[id]
            if !noComputation(node)
                str *= indent_str * "    " * call_gs(code, node)
                if code.parallel
                    str *= indent_str * @sprintf("    notify(%s)\n", nodeValid(code, id))
                else
                    str *= indent_str * @sprintf("    %s=true\n", nodeValid(code, id))
                end
            end
        end
        # return command
        str *= indent_str * "    return\n"
        str *= indent_str * "end\n"
    end
    code.computes *= str
end

export expression!
function expression!(
    code::Code,
    expr::Expr
)
    code.expressions *= string(expr)
end