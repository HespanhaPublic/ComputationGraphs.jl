export denseChain!

"""
    (; inference, training, theta) = denseChain!(graph; 
        nNodes, inferenceBatchSize, trainingBatchSize, activation,loss)

Recipe used construct a graph for inference and training of a dense forward neural network.

```
    x[1]   = input
    x[2]   = activation(W[1] * x[1] + b[1])
    ...
    x[N-1] = activation(W[N-2] * x[N-2] + b[N-2])
    output = W[N-1] * x[N-1] + b[N-1]              # no activation in the last layer
```

with a loss function of

```
  loss = lossFunction(output-reference)
```
    
# Parameters:
+ `graph::ComputationGraph{TypeValue}`; Computation graph that is updated "in-place" by adding to it
        all the nodes needed to perform one step of gradient descent.

+ `nNodes::Vector{Int}`: Vector with the number of nodes in each layer, starting from the
        input and ending at the output layer.

+ `inferenceBatchSize::Int=1`: Number of inputs for each inference batch.
        When `inferenceBatchSize=0` no nodes will be created for inference.

+ `trainingBatchSize::Int=0`: Number of inputs for each training batch. 
        When `trainingBatchSize=0` no nodes will be created for training.

+ `activation::Function=ComputationGraphs.relu`: Activation function. 
        Use the `identity` function if no activation is desired.

+ `loss::Symbol=:mse`: Desired type of loss function, among the options:

        + :sse = sum of square error
        + :mse = mean-square error (i.e., sse normalized by the error size)
        + :huber = huber function on the error
        + :mhuber = huber function on the error, normalized by the error size

# Returns: named tuple with the following fields

+ `inference::NamedTuple`: named tuple with the inference nodes:

        + `input` NN input for inference
        + `output` NN output for inference
        When `inferenceBatchSize=0` this tuple is returned empty

`+ training::NamedTuple`: named tuple with the training nodes:
        + `input` NN input for training
        + `output` NN output for training
        + `reference` NN desired output for training
        + `loss` NN loss for training

        When `trainingBatchSize=0` this tuple is returned empty

+ `theta::NamedTuple`: named tuple with the NN parameters (all the matrices W and b)

# Example:

```julia
using ComputationGraphs, Random
graph=ComputationGraph{Float32}()
(; inference, training, theta)=denseChain!(graph; 
        nNodes=[1,20,20,20,2], inferenceBatchSize=1, trainingBatchSize=3,
        activation=ComputationGraphs.relu, loss=:mse)

# (repeatable) random initialization of the weights
Random.seed!(0)
for k in eachindex(theta)
    set!(graph,theta[k],randn(Float32,size(theta[k])))
end

# Compute output for a random input
input=randn(Float32,size(inference.input))
set!(graph,inference.input,input)
output=get(graph,inference.output)
println("input = ",input,", output = ",output)

# compute loss for a batch of random inputs and desired outputs (reference)
input=randn(Float32,size(training.input))
reference=randn(Float32,size(training.reference))
set!(graph,training.input,input)
set!(graph,training.reference,reference)
loss=get(graph,training.loss)
println("inputs = ",input,", loss = ",loss)
```
"""
function denseChain!(
    graph::ComputationGraph{TypeValue};
    nNodes::Vector{Int}=Int[],
    W::Vector{TypeArray{TypeValue,2}}=TypeArray{TypeValue,2}[],
    b::Vector{TypeArray{TypeValue,1}}=TypeArray{TypeValue,1}[],
    inferenceBatchSize::Int=1,
    trainingBatchSize::Int=0,
    activation::Function=ComputationGraphs.relu,
    loss::Symbol=:mse,
    code::Union{Code,Nothing}=nothing,
) where {TypeValue}
    # NN parameters
    if isempty(nNodes)
        @assert !isempty(W) "nNodes is empty and cannot be deduced from W"
        nNodes = [size(Wk, 2) for Wk in W]
        push!(nNodes, size(W[end], 1))
    end

    nLayers = length(nNodes)
    inputSize = nNodes[1]
    outputSize = nNodes[end]
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

    # NN inputs
    if inferenceBatchSize > 1
        inferenceInput = variable(graph, inputSize, inferenceBatchSize)
    elseif inferenceBatchSize > 0
        inferenceInput = variable(graph, inputSize)
    end
    if trainingBatchSize > 1
        trainingInput = variable(graph, inputSize, trainingBatchSize)
        trainingReference = variable(graph, outputSize, trainingBatchSize)
    elseif trainingBatchSize > 0
        trainingInput = variable(graph, inputSize)
        trainingReference = variable(graph, outputSize)
    end

    ## Build chain
    if inferenceBatchSize > 0
        inferenceX = inferenceInput
        if inferenceBatchSize > 1
            inferenceOnes = ones(graph, inferenceBatchSize)
        end
    end
    if trainingBatchSize > 0
        trainingX = trainingInput
        if trainingBatchSize > 1
            trainingOnes = ones(graph, trainingBatchSize)
        end
    end
    for k in 1:nMatrices
        if inferenceBatchSize > 0
            # affine transformation
            # FIXME product by ones should be handled by Symbolic Simplifications
            #bb = (inferenceBatchSize > 1) ? timesAdjoint(graph, b[k], inferenceOnes) : b[k]
            #inferenceX = affine(graph, W[k], inferenceX, bb)
            inferenceX = affine(graph, W[k], inferenceX, b[k])
            if k < nMatrices
                inferenceX = activation(graph, inferenceX)
            end
        end
        if trainingBatchSize > 0
            # affine transformation
            # FIXME product by ones should be handled by Symbolic Simplifications
            #bb = (trainingBatchSize > 1) ? timesAdjoint(graph, b[k], trainingOnes) : b[k]
            #trainingX = @add graph affine(W[k], trainingX, bb)
            trainingX = @add graph affine(W[k], trainingX, b[k])
            if k < nMatrices
                trainingX = activation(graph, trainingX)
            end
        end
    end
    if inferenceBatchSize > 0
        inferenceOutput = inferenceX
    end
    if trainingBatchSize > 0
        trainingOutput = trainingX
        difference = @add graph trainingOutput - trainingReference
        len::TypeValue = length(difference)
        if loss == :sse
            trainingLoss = norm2(graph, difference)
        elseif loss == :mse
            trainingLoss = norm2(graph, difference)
            trainingLoss = @add graph divideScalar(trainingLoss, constant(len))
        elseif loss == :huber
            trainingLoss = huber(graph, difference)
        elseif loss == :mhuber
            trainingLoss = huber(graph, difference)
            trainingLoss = @add graph divideScalar(trainingLoss, constant(len))
        else
            error("Unknown loss function: $loss")
        end
    end

    # network parameters
    theta = (;
        (Symbol(:layer, k, :_W) => W[k] for k in 1:nMatrices)...,
        (Symbol(:layer, k, :_b) => b[k] for k in 1:nMatrices)...)

    if inferenceBatchSize > 0
        # inference nodes
        nOps = (;
            sum_output=inferenceBatchSize * sum(size(W[k], 2) * size(W[k], 1)
                                                for k in 1:nMatrices),
            prod_output=inferenceBatchSize * sum(size(W[k], 2) * size(W[k], 1)
                                                 for k in 1:nMatrices),
            relu_output=inferenceBatchSize * sum(size(W[k], 2) for k in 1:nMatrices-1))

        inference = (input=inferenceInput, output=inferenceOutput, nOps)
    else
        inference = ()
    end
    if trainingBatchSize > 0
        # training nodes
        training = (input=trainingInput, output=trainingOutput,
            reference=trainingReference, loss=trainingLoss)
    else
        training = ()
    end

    if !isnothing(code)
        ## Add code
        sets!(code, theta => "setTheta!")
        if inferenceBatchSize > 0
            sets!(code, inference.input => "setInferenceInput!")
            gets!(code, inference.output => "getInferenceOutput")
        end
        if trainingBatchSize > 0
            sets!(code,
                training.input => "setTrainingInput!",
                training.reference => "setTrainingOutput!",
            )
            gets!(code, training.loss => "getTrainingLoss")
        end
    end

    return (; theta, inference, training,)
end