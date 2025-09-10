using ProgressMeter
using ConvergenceLoggers
using Unrolled
using TimerOutputs

abstract type Optimizer end

###################
## Build Forward NN
###################

export denseChain
"""
    denseChain(TypeValue;
        nNodes=[],
        W=TypeArray{TypeValue,2}[],
        b=TypeArray{TypeValue,1}[],
        trainingBatchSize,
        inferenceBatchSize,
        activation=ComputationGraphs.relu,
        loss::Symbol=:sse,
        optimizer=NoOptimizer(),
        includeGradients=false,
        codeName="",
        parallel=false
    ) 

Create computation graph for a dense forward neural network, defined as follows:

    ```
    x[1]   = input

    z[k]   = W[k] * x[k] + b[k]    for k in 1,...,K

    x[k+1] = activation(z[k])            for k in 1,...,K-1

    output = z[K]

    loss = norm2(output-desiredOutput)

    g[loss,W[k]] = gradient(loss,W[k]) for k in 1,...,K

    g[loss,b[k]] = gradient(loss,b[k]) for k in 1,...,K
    ```

# Parameters:
+ `::Type{TypeValue}`: default type for the values of the computation graph nodes
+ `nNodes::Vector{Int}=Int[]`: vector with the number of nodes in each layer, starting from the
        input and ending at the output layer.
+ `W::Vector{TypeArray{TypeValue,2}}=TypeArray{TypeValue,2}[]`:
+ `b::Vector{TypeArray{TypeValue,1}}=TypeArray{TypeValue,1}[]`:
+ `trainingBatchSize::Int`:
+ `inferenceBatchSize::Int`:
+ `activation::Function=ComputationGraphs.relu`:
+ `loss::Symbol=:sse`:
+ `optimizer::Op=NoOptimizer()`:
+ `includeGradients::Bool=false`:
+ `codeName::String=""`:
+ `parallel::Bool=false`:

# Returns: Named tuple with fields
+ `graph`
+ `ioNodes` 
+ `parameterNodes` 
+ `trainingNodes`
+ `optimizerNodes`
+ `code`
+ `nOpsI2O`

# Number of forward operations to compute `output`:

+ z[k]: 
    + \\# prods = `sum(size(W[k],2)*(size(W[k],1)) for k in 1:K)`
    + \\# sums  = `sum(size(W[k],2)*(size(W[k],1)) for k in 1:K)`

+ x[k+1]:
    + \\# activation = `sum(size(W[k],2) for k in 1:K-1)`
"""
function denseChain(
    ::Type{TypeValue};
    nNodes::Vector{Int}=Int[],
    W::Vector{TypeArray{TypeValue,2}}=TypeArray{TypeValue,2}[],
    b::Vector{TypeArray{TypeValue,1}}=TypeArray{TypeValue,1}[],
    inferenceBatchSize::Int=1,
    trainingBatchSize::Int=0,
    activation::Function=ComputationGraphs.relu,
    loss::Symbol=:sse,
    optimizer::Symbol=:none,
    codeName::String="",
    parallel::Bool=false
) where {TypeValue}
    ## Start with empty graph
    graph = ComputationGraph{TypeValue}()

    # generate NN code
    if !isempty(codeName)
        code = Code(graph, codeName; type=:struct, unrolled=false, parallel)
    else
        code = nothing
    end

    # add NN
    (; theta, inference, training,) = denseChain!(graph; nNodes, W, b, inferenceBatchSize, trainingBatchSize, activation, loss, code)

    if isempty(W)
        W = [randn(TypeValue, nNodes[k+1], nNodes[k]) for k in 1:length(nNodes)-1]
    end
    if isempty(b)
        b = [randn(TypeValue, nNodes[k+1]) for k in 1:1:length(nNodes)-1]
    end

    # add Optimizer
    if optimizer == :none
        optimizerNodes = ()
    elseif optimizer == :gradDescent
        optimizerNodes = gradDescent!(graph; training.loss, theta)
    elseif optimizer == :adam
        optimizerNodes = adam!(graph; training.loss, theta)
    else
        error("unknown optimizer")
    end

    return (; graph,
        theta, inference, training,
        optimizerNodes...,
        code,
    )
end

################
## Flux networks
################

using Flux, ChainRulesCore
export denseChain_FluxZygote
"""
Construct dense forward neural network using Flux+Zygote
"""
function denseChain_FluxZygote(
    ::Type{TypeValue};
    W::Vector{TypeArray{TypeValue,2}}=TypeArray{TypeValue,2}[],
    b::Vector{TypeArray{TypeValue,1}}=TypeArray{TypeValue,1}[],
    activation=Flux.relu,
    loss::Symbol=:sse,
) where {TypeValue}
    println("denseChain_FluxZygote:")
    model = Chain(
        [Dense(W[k], b[k], activation) for k in 1:length(W)-1]...,
        Dense(W[end], b[end]),
    )
    display(model)
    @show loss
    loss_fun = missing
    if loss == :sse
        loss_fun = (model, input, desiredOutput) ->
            sum(abs2, model(ignore_derivatives(input)) - ignore_derivatives(desiredOutput))

    elseif loss == :mse
        loss_fun = (model, input, desiredOutput) ->
            Flux.mse(model(ignore_derivatives(input)), ignore_derivatives(desiredOutput))

    else
        error("Unknown loss function: $loss")
    end

    loss_grad_fun = (model, input, desiredOutput) -> Flux.withgradient(loss_fun, model, input, desiredOutput)
    return (
        model,
        loss_fun,
        loss_grad_fun,
    )
end

using Flux, Enzyme
export denseChain_FluxEnzyme
"""
Construct dense forward neural network using Flux+Enzyme
"""
function denseChain_FluxEnzyme(
    ::Type{TypeValue};
    W::Vector{TypeArray{TypeValue,2}}=TypeArray{TypeValue,2}[],
    b::Vector{TypeArray{TypeValue,1}}=TypeArray{TypeValue,1}[],
    activation=Flux.relu,
    loss::Symbol=:sse,
) where {TypeValue}
    println("denseChain_FluxZygote:")
    model = Chain(
        [Dense(W[k], b[k], activation) for k in 1:length(W)-1]...,
        Dense(W[end], b[end]),
    )
    display(model)
    @show loss
    loss_fun = missing
    if loss == :sse
        loss_fun = (model, input, desiredOutput) -> sum(abs2, model(input) - desiredOutput)
    elseif loss == :mse
        loss_fun = (model, input, desiredOutput) -> Flux.mse(model(input), desiredOutput)
    else
        error("Unknown loss function: $loss")
    end

    loss_grad_fun = (model, input, desiredOutput) -> Flux.withgradient(loss_fun,
        Duplicated(model), Const(input), Const(desiredOutput))
    return (;
        model,
        loss_fun,
        loss_grad_fun,
    )
end



######################
## Build Q-learning NN
######################

export denseQlearningChain
"""
Create computation graph for a dense forward neural network used to store reinforcement learning's Q-function.
"""
function denseQlearningChain(
    ::Type{TypeValue};
    nNodes::Vector{Int}=Int[],
    W::Vector{TypeArray{TypeValue,2}}=TypeArray{TypeValue,2}[],
    b::Vector{TypeArray{TypeValue,1}}=TypeArray{TypeValue,1}[],
    inferenceBatchSize::Int,
    trainingBatchSize::Int,
    activation::Function=ComputationGraphs.relu,
    loss::Symbol=:sse,
    optimizer::Symbol=:none,
    codeName::String="",
    parallel::Bool=false
) where {TypeValue}
    ## Start with empty graph
    graph = ComputationGraph{TypeValue}()

    # FIXME: a lot of this code is duplicated with RecipesNN.denseChain!(), should unify

    # generate NN code
    if !isempty(codeName)
        code = Code(graph, codeName; type=:struct, unrolled=false, parallel)
    else
        code = nothing
    end

    if isempty(nNodes)
        @assert !isempty(W) "nNodes is empty and cannot be deduced from W"
        nNodes = [size(Wk, 2) for Wk in W]
        push!(nNodes, size(W[end], 1))
    end

    nLayers = length(nNodes)
    nMatrices = length(nNodes) - 1

    @assert nLayers >= 2 "NN must have at least 2 layers (input and output)"

    # make W and b variables
    if isempty(W)
        W = Tuple(variable(graph, nNodes[k+1], nNodes[k]) for k in 1:nMatrices)
    else
        W = Tuple(variable(graph, Wk) for Wk in W)
    end
    if isempty(b)
        b = Tuple(variable(graph, nNodes[k+1]) for k in 1:nMatrices)
    else
        b = Tuple(variable(graph, bk) for bk in b)
    end

    @assert length(nNodes) == length(W) + 1 error("mismatch between nNodes and W")
    for k in 1:nMatrices
        @assert size(W[k]) == (nNodes[k+1], nNodes[k])
    end
    @assert length(nNodes) == length(b) + 1 error("mismatch between nNodes and b")
    for k in 1:nMatrices
        @assert size(b[k]) == (nNodes[k+1],)
    end

    nInputs = nNodes[1]
    nOutputs = nNodes[end]

    # inputs
    inferenceState = variable(graph, nInputs, inferenceBatchSize)
    trainingState = variable(graph, nInputs, trainingBatchSize)
    trainingNextState = variable(graph, nInputs, trainingBatchSize)
    trainingReward = variable(graph, trainingBatchSize)
    trainingSign = variable(graph, trainingBatchSize)
    trainingAction = variable(graph, Int, Vector{Int}(undef, trainingBatchSize))# give type to avoid warning

    # chain
    inferenceX = inferenceState
    trainingX = trainingState
    trainingNextX = trainingNextState
    inferenceOnes = ones(graph, inferenceBatchSize)
    if trainingBatchSize > 0
        trainingOnes = ones(graph, trainingBatchSize)
    end
    for k in 1:nMatrices
        #bb = timesAdjoint(graph, b[k], inferenceOnes)
        #inferenceX = affine(graph, W[k], inferenceX, bb)
        inferenceX = affine(graph, W[k], inferenceX, b[k])
        if k < nMatrices
            inferenceX = activation(graph, inferenceX)
        end
        if trainingBatchSize > 0
            #bb = timesAdjoint(graph, b[k], trainingOnes)
            #trainingX = affine(graph, W[k], trainingX, bb)
            #trainingNextX = affine(graph, W[k], trainingNextX, bb)
            trainingX = affine(graph, W[k], trainingX, b[k])
            trainingNextX = affine(graph, W[k], trainingNextX, b[k])
            if k < nMatrices
                trainingX = activation(graph, trainingX)
                trainingNextX = activation(graph, trainingNextX)
            end
        end
    end
    rhs = selectRows(graph, trainingX, trainingAction)
    inferenceOutput = inferenceX
    bestQ = maxRow(graph, trainingNextX)
    bestSignedQ = pointTimes(graph, trainingSign, bestQ)
    lhs = +(graph, trainingReward, bestSignedQ)
    difference = -(graph, rhs, lhs)
    len::TypeValue = length(difference)
    if loss == :sse
        loss = norm2(graph, difference)
    elseif loss == :mse
        loss = norm2(graph, difference)
        loss = @add graph divideScalar(loss, constant(len))
    elseif loss == :huber
        loss = huber(graph, difference)
    elseif loss == :mhuber
        loss = huber(graph, difference)
        loss = @add graph divideScalar(loss, constant(len))
    else
        error("Unknown loss function: $loss")
    end

    # network parameters
    theta = (;
        (Symbol(:layer, k, :_W) => W[k] for k in 1:nMatrices)...,
        (Symbol(:layer, k, :_b) => b[k] for k in 1:nMatrices)...)

    # inference nodes
    inference = (state=inferenceState, output=inferenceOutput)

    if trainingBatchSize > 0
        # training nodes
        training = (;
            state=trainingState, nextState=trainingNextState,
            reward=trainingReward, sign=trainingSign, action=trainingAction,
            loss)
    else
        training = ()
    end

    # add Optimizer
    if optimizer == :none
        optimizerNodes = ()
    elseif optimizer == :gradDescent
        optimizerNodes = gradDescent!(graph; training.loss, theta)
    elseif optimizer == :adam
        optimizerNodes = adam!(graph; training.loss, theta)
    else
        error("unknown optimizer")
    end

    if !isempty(codeName)
        code = Code(graph, codeName; type=:struct, unrolled=false, parallel)
        ## Add code
        sets!(code,
            theta => "setTheta!",
            inference.state => "setState!",
        )
        sets!(code,
            training.state => "setTrainingState!",
            training.nextState => "setTrainingNextState!",
            training.reward => "setTrainingReward!",
            training.sign => "setTrainingSign!",
            training.action => "setTrainingAction!",
        )
        gets!(code,
            inference.output => "getOutput",
            training.loss => "getLoss",
        )
    else
        code = nothing
    end

    return (; graph, theta, inference, training, optimizerNodes, code,)

end


#####################################
## Train NNs - given ComputationGraph
#####################################

export trainNN!, trainQlearning!

function trainNN!(;
    graph::ComputationGraph{TypeValue},
    theta,
    inference,
    training,
    optimizerNodes,
    adam_eta::TypeValue,
    adam_beta1::TypeValue=TypeValue(0.9),
    adam_beta2::TypeValue=TypeValue(0.999),
    adam_epsilon::TypeValue=TypeValue(1e-8),
    dataLoader!::Function,
    nIterations::Int,
    plots, subplot::Int
) where {TypeValue}

    timerOutput = TimerOutput()

    @timeit timerOutput "init optimizer" begin
        # set Adam parameters
        set!(graph, optimizerNodes.eta, adam_eta)
        set!(graph, optimizerNodes.beta1, adam_beta1)
        set!(graph, optimizerNodes.beta2, adam_beta2)
        set!(graph, optimizerNodes.epsilon, adam_epsilon)
        # initialize moments
        copyto!(graph, optimizerNodes.state, optimizerNodes.init_state)
    end

    @timeit timerOutput "init inference" begin
        # just to be able to do a compute all
        inferenceInput::Matrix{TypeValue} = zeros(TypeValue, size(inference.input))
        set!(graph, inference.input, inferenceInput)
    end

    @timeit timerOutput "init batch" begin
        trainingInput::Matrix{TypeValue} = similar(training.input)
        trainingOutput::Matrix{TypeValue} = similar(training.reference)
        trainingNodes = (training.input, training.reference)
        trainingValues = (trainingInput, trainingOutput)
        dataLoader!(trainingValues...)
        set!(graph, trainingNodes, trainingValues)
    end

    loss = get(graph, training.loss)
    iter = get(graph, optimizerNodes.state.iteration)

    @timeit timerOutput "init compute" begin
        @printf("Initial loss = %15.8f (iteration = %9d, max count = %9d)\n",
            loss[1], iter[1], maximum(graph.count))
    end

    # Start loggers
    lossLogger = TimeSeriesLogger{Int64,TypeValue}(1;
        yaxis=:log10, legend=[""], ylabel="loss", xlabel="iteration")
    tNext = time_ns()

    progress = Progress(nIterations; showspeed=true, dt=5, barlen=6)

    # prepare for update
    current = (; optimizerNodes.state..., theta...)
    next = (; optimizerNodes.next_state..., optimizerNodes.next_theta...)
    #current = values(current)
    #next = values(next)
    @timeit timerOutput "iteration" for iteration::Int in 1:nIterations

        if iteration > 1
            # new set of training data
            dataLoader!(trainingValues...)
            @timeit timerOutput "set batch" set!(graph, trainingNodes, trainingValues)

        end

        if true
            @timeit timerOutput "compute!(state+)" compute!(graph, next)
            @timeit timerOutput "compute!(loss)" compute!(graph, training.loss)
        elseif false
            @timeit timerOutput "compute!(all)" compute!(graph)
        end

        # optimizer update 
        @timeit timerOutput "copyto(theta+)" copyto!(graph, current, next)

        # Log losses
        @timeit timerOutput "log" if mod(iteration, 1000) == 1 || iteration == nIterations
            append!(lossLogger, iteration, [loss[1]])
            if time_ns() > tNext || iteration == nIterations
                tNext += 5e9 # 5 sec display period
                plotLogger!(plots, [subplot], [lossLogger])
                display(plots)
            end
        end

        # Update progress meter
        ProgressMeter.update!(progress, iteration;
            desc=@sprintf("iter=%10d loss=%.5f", iteration, loss[1]))
    end
    @timeit timerOutput "compute!(loss)" compute!(graph, training.loss)
    @printf("Final loss   = %15.8f (iteration = %9d, max count = %9d)\n",
        loss[1], iter[1], maximum(graph.count))
    if nIterations > 10
        print_timer(timerOutput; sortby=:firstexec)
    end
end


function trainQlearning!(;
    graph::ComputationGraph{TypeValue},
    theta,
    inference,
    training,
    optimizerNodes,
    adam_eta::TypeValue,
    adam_beta1::TypeValue=TypeValue(0.9),
    adam_beta2::TypeValue=TypeValue(0.999),
    adam_epsilon::TypeValue=TypeValue(1e-8),
    dataLoader!::Function,
    parallel4training::Bool=false,
    nIterations::Int,
    plots, subplot::Int
) where {TypeValue}

    timerOutput = TimerOutput()

    @timeit timerOutput "init optimizer" begin
        # set Adam parameters
        set!(graph, optimizerNodes.eta, adam_eta)
        set!(graph, optimizerNodes.beta1, adam_beta1)
        set!(graph, optimizerNodes.beta2, adam_beta2)
        set!(graph, optimizerNodes.epsilon, adam_epsilon)
        # initialize moments
        copyto!(graph, optimizerNodes.state, optimizerNodes.init_state)
        # initialize moments
    end

    @timeit timerOutput "init inference" begin
        # just to be able to do a compute all
        inferenceState::Matrix{TypeValue} = zeros(TypeValue, size(inference.state))
        set!(graph, inference.state, inferenceState)
    end

    @timeit timerOutput "init batch" begin
        trainingStateValues::Matrix{TypeValue} = zeros(TypeValue, size(training.state))
        trainingNextStateValues::Matrix{TypeValue} = zeros(TypeValue, size(training.nextState))
        trainingRewardValues::Vector{TypeValue} = zeros(TypeValue, size(training.reward))
        trainingSignValues::Vector{TypeValue} = zeros(TypeValue, size(training.sign))
        trainingActionValues::Vector{Int} = ones(Int, size(training.action)) # must be valid action
        trainingNodes = (
            training.state, training.action,
            training.nextState, training.reward,
            training.sign,)
        trainingValues = (
            trainingStateValues, trainingActionValues,
            trainingNextStateValues, trainingRewardValues, trainingSignValues,)
        # dummy values for data loader not to fail due to missing variable
        set!(graph, trainingNodes, trainingValues)
    end

    # Start loggers
    lossLogger = TimeSeriesLogger{Int64,TypeValue}(1;
        yaxis=:log10, legend=[""], ylabel="loss", xlabel="iteration")
    tNext = time_ns()

    progress = Progress(nIterations; showspeed=true, dt=5, barlen=6)

    if parallel4training
        computeSpawn!(graph)
    end

    @timeit timerOutput "init batch" begin
        # must appear after spawn for parallel case
        dataLoader!(trainingValues...)
        set!(graph, trainingNodes, trainingValues)
    end

    loss = get(graph, training.loss)
    iter = get(graph, optimizerNodes.state.iteration)

    @timeit timerOutput "init compute" begin
        @printf("Initial loss = %15.8f (iteration = %9d, max count = %9d)\n",
            loss[1], iter[1], maximum(graph.count)) # allocations
    end

    # prepare for update
    current = (; optimizerNodes.state..., theta...)
    next = (; optimizerNodes.next_state..., optimizerNodes.next_theta...)
    #current = values(current)
    #next = values(next)
    @timeit timerOutput "iteration" for iteration::Int in 1:nIterations

        if iteration > 1
            # new set of training data
            @timeit timerOutput "get batch" dataLoader!(trainingValues...)
            @timeit timerOutput "set batch" set!(graph, trainingNodes, trainingValues)
        end

        if parallel4training
            @timeit timerOutput "sync&request" begin
                syncValid(graph)
                request(graph, next)
                request(graph, training.loss)
            end
            @timeit timerOutput "wait(next,loss)" begin
                wait(graph, next)
                wait(graph, training.loss)
            end
        else
            @timeit timerOutput "compute!(state+)" compute!(graph, next)
            @timeit timerOutput "compute!(loss)" compute!(graph, training.loss)
        end

        # optimizer update 
        @timeit timerOutput "copyto(theta+)" copyto!(graph, current, next)

        # Log losses
        @timeit timerOutput "log" if mod(iteration, 1000) == 1 || iteration == nIterations
            append!(lossLogger, iteration, [loss[1]])
            if time_ns() > tNext || iteration == nIterations
                tNext += 5e9 # 5 sec display period
                plotLogger!(plots, [subplot], [lossLogger])
                display(plots)
            end
        end

        # Update progress meter
        ProgressMeter.update!(progress, iteration;
            desc=@sprintf("iter=%10d loss=%.5f", iteration, loss[1]))
    end
    if parallel4training
        computeUnspawn!(graph)
    end
    @timeit timerOutput "compute!(loss)" compute!(graph, training.loss)
    @printf("Final loss   = %15.8f (iteration = %9d, max count = %9d)\n",
        loss[1], iter[1], maximum(graph.count))
    if nIterations > 10
        print_timer(timerOutput; sortby=:firstexec)
    end
end

