
export compute!

###############################################
## Generate node-specific computation functions
###############################################

"""
Generates a function that conditionally evaluates a node, 
    using closure & enforcing type stability.

Each function will
1) check if each parent need to be re-evaluated, if re-evaluates the parent and sets it's valid bit
   to true.
2) always recomputes the function
    + without checking if it is needed (this should be checked by caller, to enable force=true)
    + without setting the valid bit), which is expected to be set by the calling function. 
"""
function generateComputeFunctions(
    graph::ComputationGraph{TypeValue},
    id::Int,
    computeFunction::Function,
    parameters::TP,
    parentIds::TPI,
    parentValues::TPV,
    value::TV
) where {TypeValue,TP<:Tuple,TPI<:Tuple,TPV<:Tuple,TV<:AbstractArray}
    if length(parentIds) == 0
        fun_node = let id0 = id,
            fun0 = computeFunction,
            val0 = value,
            pars = parameters,
            logCounts = graph.count,
            logTimes = graph.time

            Base.Experimental.@opaque () -> begin
                t0 = time_ns()
                fun0(val0, pars...)
                logTimes[id0] += (time_ns() - t0)
                logCounts[id0] += 1
                nothing
            end
        end
        fun_with_ancestors = fun_node
    elseif length(parentIds) == 1
        fun_node = let id0 = id,
            fun0 = computeFunction,
            val0 = value,
            val1 = parentValues[1],
            pars = parameters,
            logCounts = graph.count,
            logTimes = graph.time

            Base.Experimental.@opaque () -> begin
                t0 = time_ns()
                fun0(val0, val1, pars...)
                logTimes[id0] += (time_ns() - t0)
                logCounts[id0] += 1
                nothing
            end
        end
        fun_with_ancestors = let id0 = id,
            fun0 = computeFunction,
            val0 = value,
            val1 = parentValues[1],
            pars = parameters,
            logCounts = graph.count,
            logTimes = graph.time,
            validValue = graph.validValue,
            pid1 = parentIds[1],
            fun1 = graph.compute_with_ancestors[pid1]

            Base.Experimental.@opaque () -> begin
                if !validValue[pid1]
                    do_ccall(fun1)
                    validValue[pid1] = true
                end
                t0 = time_ns()
                fun0(val0, val1, pars...)
                logTimes[id0] += (time_ns() - t0)
                logCounts[id0] += 1
                nothing
            end
        end
    elseif length(parentIds) == 2
        fun_node = let id0 = id,
            fun0 = computeFunction,
            val0 = value,
            val1 = parentValues[1],
            val2 = parentValues[2],
            pars = parameters,
            logCounts = graph.count,
            logTimes = graph.time

            Base.Experimental.@opaque () -> begin
                t0 = time_ns()
                fun0(val0, val1, val2, pars...)
                logTimes[id0] += (time_ns() - t0)
                logCounts[id0] += 1
                nothing
            end
        end
        fun_with_ancestors = let id0 = id,
            fun0 = computeFunction,
            val0 = value,
            val1 = parentValues[1],
            val2 = parentValues[2],
            pars = parameters,
            logCounts = graph.count,
            logTimes = graph.time,
            validValue = graph.validValue,
            pid1 = parentIds[1],
            pid2 = parentIds[2],
            fun1 = graph.compute_with_ancestors[pid1],
            fun2 = graph.compute_with_ancestors[pid2]

            Base.Experimental.@opaque () -> begin
                if !validValue[pid1]
                    do_ccall(fun1)
                    validValue[pid1] = true
                end
                if !validValue[pid2]
                    do_ccall(fun2)
                    validValue[pid2] = true
                end
                t0 = time_ns()
                fun0(val0, val1, val2, pars...)
                logTimes[id0] += (time_ns() - t0)
                logCounts[id0] += 1
                nothing
            end
        end
    elseif length(parentIds) == 3
        fun_node = let id0 = id,
            fun0 = computeFunction,
            val0 = value,
            val1 = parentValues[1],
            val2 = parentValues[2],
            val3 = parentValues[3],
            pars = parameters,
            logCounts = graph.count,
            logTimes = graph.time

            Base.Experimental.@opaque () -> begin
                t0 = time_ns()
                fun0(val0, val1, val2, val3, pars...)
                logTimes[id0] += (time_ns() - t0)
                logCounts[id0] += 1
                nothing
            end
        end
        fun_with_ancestors = let id0 = id,
            fun0 = computeFunction,
            val0 = value,
            val1 = parentValues[1],
            val2 = parentValues[2],
            val3 = parentValues[3],
            pars = parameters,
            logCounts = graph.count,
            logTimes = graph.time,
            validValue = graph.validValue,
            pid1 = parentIds[1],
            pid2 = parentIds[2],
            pid3 = parentIds[3],
            fun1 = graph.compute_with_ancestors[pid1],
            fun2 = graph.compute_with_ancestors[pid2],
            fun3 = graph.compute_with_ancestors[pid3]

            Base.Experimental.@opaque () -> begin
                if !validValue[pid1]
                    do_ccall(fun1)
                    validValue[pid1] = true
                end
                if !validValue[pid2]
                    do_ccall(fun2)
                    validValue[pid2] = true
                end
                if !validValue[pid3]
                    do_ccall(fun3)
                    validValue[pid3] = true
                end
                t0 = time_ns()
                fun0(val0, val1, val2, val3, pars...)
                logTimes[id0] += (time_ns() - t0)
                logCounts[id0] += 1
                nothing
            end
        end
    elseif length(parentIds) == 4 # currently only used by affineRows
        fun_node = let id0 = id,
            fun0 = computeFunction,
            val0 = value,
            val1 = parentValues[1],
            val2 = parentValues[2],
            val3 = parentValues[3],
            val4 = parentValues[4],
            pars = parameters,
            logCounts = graph.count,
            logTimes = graph.time

            Base.Experimental.@opaque () -> begin
                t0 = time_ns()
                fun0(val0, val1, val2, val3, val4, pars...)
                logTimes[id0] += (time_ns() - t0)
                logCounts[id0] += 1
                nothing
            end
        end
        fun_with_ancestors = let id0 = id,
            fun0 = computeFunction,
            val0 = value,
            val1 = parentValues[1],
            val2 = parentValues[2],
            val3 = parentValues[3],
            val4 = parentValues[4],
            pars = parameters,
            logCounts = graph.count,
            logTimes = graph.time,
            validValue = graph.validValue,
            pid1 = parentIds[1],
            pid2 = parentIds[2],
            pid3 = parentIds[3],
            pid4 = parentIds[4],
            fun1 = graph.compute_with_ancestors[pid1],
            fun2 = graph.compute_with_ancestors[pid2],
            fun3 = graph.compute_with_ancestors[pid3],
            fun4 = graph.compute_with_ancestors[pid4]

            Base.Experimental.@opaque () -> begin
                if !validValue[pid1]
                    do_ccall(fun1)
                    validValue[pid1] = true
                end
                if !validValue[pid2]
                    do_ccall(fun2)
                    validValue[pid2] = true
                end
                if !validValue[pid3]
                    do_ccall(fun3)
                    validValue[pid3] = true
                end
                if !validValue[pid4]
                    do_ccall(fun4)
                    validValue[pid4] = true
                end
                t0 = time_ns()
                fun0(val0, val1, val2, val3, val4, pars...)
                logTimes[id0] += (time_ns() - t0)
                logCounts[id0] += 1
                nothing
            end
        end
    else
        error()
    end

    # using a simplified version of FunctionWrappers.jl
    compute_with_ancestors = FunctionWrapper(fun_with_ancestors)
    compute_node = FunctionWrapper(fun_node)

    return (compute_with_ancestors, compute_node)
end

###############################################################
## Direct calls to files generated by `generateComputeFunction`
###############################################################

"""
    compute_node!(node)
    compute_node!(graph,node)
    compute_node!(graph,id)

Call the function generated by `generateComputeFunction` that computes a single node.
"""
@inline function compute_node!(
    node::Node
) where {Node<:AbstractNode}
    do_ccall(node.compute!)
end
@inline compute_node!(
    ::ComputationGraph{TypeValue},
    node::Node
) where {TypeValue,Node<:AbstractNode} = compute_node!(node)
@inline compute_node!(
    graph::ComputationGraph{TypeValue},
    id::Int
) where {TypeValue} = compute_node!(graph.nodes[id])

"""
    compute_with_ancestors!(node)
    compute_with_ancestors!(graph,node)
    compute_with_ancestors!(graph,id)

Call the function generated by `generateComputeFunction` that computes a node and all its required
parents.
"""
@inline compute_with_ancestors!(
    graph::ComputationGraph{TypeValue},
    node::Node
) where {TypeValue,Node<:AbstractNode} = compute_with_ancestors!(graph, node.id)
@inline compute_with_ancestors!(
    graph::ComputationGraph{TypeValue},
    id::Int
) where {TypeValue} = do_ccall(graph.compute_with_ancestors[id])

############################
## Recompute the whole graph
############################

"""
Recompute the whole graph
"""
function compute!(
    graph::ComputationGraph{TypeValue};
    force::Bool=false
) where {TypeValue}
    if force
        for (id, node) in enumerate(graph.nodes)
            # invalid (uninitialized) variables must call compute to give error
            if isa(node, NodeVariable)
                if !graph.validValue[id]
                    error("trying to compute a graph with variable node $(id) not initialized")
                end
            elseif !isa(node, AbstractConstantNode)  # constant node cannot be computed
                compute_node!(node)
            end
        end
        fill!(graph.validValue, true)
    else
        # evaluating nodes "in order", so no need to worry about ancestors
        for (id, valid) in enumerate(graph.validValue)
            if !valid
                node = graph.nodes[id]
                compute_node!(node)
                graph.validValue[id] = true
            end
        end
    end
    return nothing
end

################################
## Recursive only what is needed
################################

"""
Recompute only what is needed to get a node
"""
function compute!(
    graph::ComputationGraph{TypeValue},
    node::Node
) where {TypeValue,Node<:AbstractNode}
    id::Int = node.id
    if !graph.validValue[id]
        compute_with_ancestors!(graph, id)
        graph.validValue[id] = true
    end
    return nothing
end

"""
Recompute only what is needed to get a vector/tuple of node
"""
@inline compute!(
    graph::ComputationGraph{TypeValue},
    nodes::NT
) where {TypeValue,NT<:NamedTuple} = compute!(graph, values(nodes))

@unroll function compute!(
    graph::ComputationGraph{TypeValue},
    nodes::NTuple{N,Any}
) where {TypeValue,N}
    @unroll for node in nodes
        id::Int = node.id
        if !graph.validValue[id]
            compute_with_ancestors!(graph, id)
            graph.validValue[id] = true
        end
    end
    return nothing
end

################
## compute + get
################

"""
Get the value of a node, performing whatever computations are needed.
"""
function Base.get(
    graph::ComputationGraph{TypeValue},
    node::Node
) where {TypeValue,Node<:AbstractNode}
    compute!(graph, node)
    value = nodeValue(node)
    return value
end

"""Get the values of a list of node."""
function Base.get(
    graph::ComputationGraph{TypeValue},
    nodes::NT
) where {TypeValue,NT<:Union{Tuple,NamedTuple}}
    compute!(graph, nodes)
    return (; (k => nodeValue(node) for (k, node) in pairs(nodes))...)
end

#######################################
## Spawn tasks for parallel computation
#######################################

export computeSpawn!, computeUnspawn!, request, wait, syncValid

"""
    computeSpawn!(graph)

Spans a set of tasks for parallel evaluation of a computation graph.
"""
function computeSpawn!(
    graph::ComputationGraph{TypeValue}
) where {TypeValue}
    computeUnspawn!(graph)
    print("Spawning computation nodes  :")

    # initialize valid events
    syncValid(graph)
    # enable all parallel flags
    fill!(graph.enableTask, true)
    # reset all request events
    reset.(graph.requestEvent)

    #@show getproperty.(graph.requestEvent, :set)

    nTasks = 0
    for (id, node) in enumerate(graph.nodes)
        graph.tasks[id] = if !noComputation(node)
            ##@printf("    spawning task for node %d (%s)\n", id, typeof(node).name.name)
            #print(" ", id)
            #print(".")
            nTasks += 1
            Threads.@spawn begin
                while true
                    # wait for request
                    #@printf("\n   node %d waiting for request (cnt=%d)\n", $id, count[$id])
                    wait($(graph.requestEvent)[$id])
                    if !$(graph.enableTask)[$id]
                        break
                    else
                    end
                    if !$(graph.validEvent)[$id].set
                        # notify parents that need computation
                        for pid in $(node.parentIds)
                            #$(graph.validEvent)[pid].set || @printf("      node %d checking parent %d\n", $id, pid)
                            $(graph.validEvent)[pid].set || notify($(graph.requestEvent)[pid])
                        end
                        # wait for parents that need computation
                        for pid in $(node.parentIds)
                            #$(graph.validEvent)[pid].set || @printf("      node %d waiting for parent %d\n", $id, pid)
                            $(graph.validEvent)[pid].set || wait($(graph.validEvent)[pid])
                        end
                        do_ccall($(node.compute!)) # already increments count
                        notify($(graph.validEvent)[$id])
                        $(graph.validValue)[$id] = true
                        #@printf("\n   node %d finished computation (cnt=%d)\n", $id, count[$id])
                    end
                end # while
                # @printf("node %d tasks exited\n", $id)
            end # @spawn
        else
            nothing
        end # if 
    end # for
    @printf(" spawn   %5d tasks\n", nTasks)
    #display(graph.tasks)
end

# TODO: needs to be automated
"""
    syncValid(graph)

Updates `graph.validEvents::Threads.Event` with `graph.validValues::BitValue`.

# Usage:

1) This function is automatically called from within [ComputationGraphs.computeSpawn!(graph)](@ref).
2) It *needs to be explicitly called* if [ComputationGraphs.set!](@ref) or [ComputationGraphs.copyto!](@ref) is called upon any variable after
   [ComputationGraphs.computeSpawn!(graph)](@ref) was issued.
"""
function syncValid(
    graph::ComputationGraph{TypeValue}
) where {TypeValue}
    for (id, node) in enumerate(graph.nodes)
        if graph.validValue[id]
            notify(graph.validEvent[id])
        else
            reset(graph.validEvent[id])
        end
        # check for valid variables
        if isa(node, NodeVariable) && !graph.validValue[id]
            display(node)
            error("Cannot spawn graph with variable node $id invalid")
        end
    end
end

"""
    computeUnspawn!(graph)

Terminates the tasks spawned by [ComputationGraphs.computeSpawn!(graph)](@ref)
"""
function computeUnspawn!(
    graph::ComputationGraph{TypeValue}
) where {TypeValue}
    print("Unspawning computation nodes:")
    # disable tasks
    fill!(graph.enableTask, false)
    # force completion
    requests = true
    nTasks = 0
    failed = 0
    while requests
        requests = false
        for (id, node) in enumerate(graph.nodes)
            if !isnothing(graph.tasks[id]) && istaskfailed(graph.tasks[id])
                failed += 1
            end
            if !isnothing(graph.tasks[id]) && istaskstarted(graph.tasks[id]) && !istaskdone(graph.tasks[id])
                requests = true
                #@printf("requesting node %d to terminate (%s)\n", id, graph.tasks[id])
                #print(" ", id)
                #print(".")
                nTasks += 1
                request(graph, node)
                yield()
                #wait(graph.tasks[id])
            else
                #if !isnothing(graph.tasks[id])
                #    @printf(" %d (%d)", id, graph.tasks[id]._state)
                #end
            end
        end
    end
    @printf(" unspawn %5d tasks (%d failed)\n", nTasks, failed)
end

"""
    request(graph, node::Node)
    request(graph, node::NTuple{Node})
    request(graph, node::NamedTuple{Node})

Requests parallel evaluation of a node or a tuple of nodes. 

Presumes a previous call to `computeSpawn!(graph)`
"""
@inline function request(graph::ComputationGraph{TypeValue}, node::Node
) where {TypeValue,Node<:AbstractNode}
    #@printf("   requesting computation by node %d\n", node.id)
    notify(graph.requestEvent[node.id])
end
@unroll function request(graph::ComputationGraph{TypeValue}, nodes::Tuple
) where {TypeValue}
    @unroll for node in nodes
        request(graph, node)
    end
end
@inline request(graph::ComputationGraph{TypeValue}, nodes::NamedTuple
) where {TypeValue} = request(graph, values(nodes))

"""
    wait(graph, node::Node)
    wait(graph, node::NTuple{Node})
    wait(graph, node::NamedTuple{Node})

Waits for the evaluation of a node or a tuple of nodes, after an appropriate computation request
made using `request(graph, node(s))`

Presumes a previous call to `computeSpawn!(graph)`
"""
@inline function Base.wait(graph::ComputationGraph{TypeValue}, node::Node
) where {TypeValue,Node<:AbstractNode}
    #@printf("   waiting for node %d\n", node.id)
    wait(graph.validEvent[node.id])
end
@unroll function Base.wait(graph::ComputationGraph{TypeValue}, nodes::Tuple
) where {TypeValue}
    @unroll for node in nodes
        wait(graph, node)
    end
end
# TODO: need to check allocations
@inline Base.wait(graph::ComputationGraph{TypeValue}, nodes::NT) where {TypeValue,NT<:NamedTuple} =
    wait(graph, values(nodes))
