using Revise

using LinearAlgebra
using LogExpFunctions

using ComputationGraphs

using Random
using BenchmarkTools
using Printf
using Term

using Test

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 3

begin
    BLAS.set_num_threads(1) # does not seem to make a difference for small size, but 1 actually faster for "large" sizes
    BLAS.set_num_threads(3)
    @show Threads.nthreads() # does not seem to make a difference 
    @show BLAS.get_num_threads()
    @show Base.JLOptions().opt_level
end

begin
    # NN Parameters
    TypeValue = Float64
    nNodes = [3, 2, 5, 4] .* 10
    Random.seed!(0)
    K = length(nNodes) - 1
    W = [randn(TypeValue, nNodes[k+1], nNodes[k]) for k in 1:K]
    b = [randn(TypeValue, nNodes[k+1]) for k in 1:K]
    # single sample
    input = randn(TypeValue, nNodes[1])
    reference = randn(TypeValue, nNodes[end])
    # batch
    trainingBatchSize = 1
    batchInput = randn(TypeValue, nNodes[1])
    batchReference = randn(TypeValue, nNodes[end])
    # error tolerance for testing
    tol = 1e-8
    activation = ComputationGraphs.relu
    activation = ComputationGraphs.logistic
end

@testset "test_Hessian: compute hessian" begin
    lossType = :sse # TODO `loss_grad_Enzyme` given error with :mse
    (graph, theta, inference, training, optimizerNodes...) = denseChain(TypeValue;
        nNodes,
        inferenceBatchSize=1,
        trainingBatchSize,
        activation,
        loss=lossType, optimizer=:gradDescent)

    @time "hessian(loss)" h_ = hessian(graph, training.loss, training.reference, training.input)

    function test_(graph::ComputationGraph)
        @time "compute graph" compute!(graph; force=true)
        for (i, node) in enumerate(graph.nodes)
            @show (i, typeof(node))
            @time "   compute node " compute_node!(node)
            a = @allocated compute_node!(node)
            @test a == 0
        end
    end
    #test_(graph)

    # initialize everything to prevent errors in compute!()
    set!(graph, values(theta), (W..., b...))
    set!(graph, inference.input, input)
    set!(graph, training.input, batchInput)
    set!(graph, training.reference, batchReference)
    set!(graph, optimizerNodes.eta, 1e-3)

    resetLog!(graph)
    bmk = @benchmark compute!($graph; force=true)
    display(bmk)
    @test bmk.allocs == 0

    @time "get(hessian)" h = [get(graph, hk) for hk in h_]

    H = reduce(hcat, h)
    @show size(H)
    @test size(H) == (length(reference), length(input))
end