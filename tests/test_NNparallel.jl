using Revise

using LinearAlgebra

using ComputationGraphs

using Random
using BenchmarkTools
using Printf
using Term

using Test

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5

begin
    # NN Parameters
    TypeValue = Float64
    nNodes = [1, 10, 10, 10, 1] .* 10
    Random.seed!(0)
    K = length(nNodes) - 1
    W = [randn(TypeValue, nNodes[k+1], nNodes[k]) for k in 1:K]
    b = [randn(TypeValue, nNodes[k+1]) for k in 1:K]
    # single sample
    input = randn(TypeValue, nNodes[1])
    desiredOutput = randn(TypeValue, nNodes[end])
    # batch
    trainingBatchSize = 2000
    batchInput = randn(TypeValue, nNodes[1], trainingBatchSize)
    batchReference = randn(TypeValue, nNodes[end], trainingBatchSize)
    # error tolerance for testing
    tol = 1e-8
end

@testset "NN: check correctness" begin
    lossType = :sse # TODO `loss_grad_Enzyme` given error with :mse
    (graph, theta, inference, training, optimizerNodes...) = denseChain(TypeValue;
        nNodes,
        inferenceBatchSize=1,
        trainingBatchSize,
        loss=lossType, optimizer=:gradDescent)

    """set all values to zero and reinitialize variables"""
    function resetGraph()
        for node in graph.nodes
            if !ComputationGraphs.noComputation(node)
                val = nodeValue(node)
                fill!(val, 0.0)
            end
        end
        set!(graph, values(theta), (W..., b...))
        set!(graph, inference.input, input)
        set!(graph, training.input, batchInput)
        set!(graph, training.reference, batchReference)
        set!(graph, optimizerNodes.eta, 1e-3)
        syncValid(graph)  # TODO: needs to be automated 

    end

    ### serial computation
    println(@bold @green "Serial computation: ")
    resetGraph()
    resetLog!(graph)
    compute!(graph, training.loss)
    compute!(graph, optimizerNodes.gradients)
    @show Int(maximum(graph.count))
    @test maximum(graph.count) == 1

    @show lossSerial = copy(get(graph, training.loss))
    gradientSerial = copy.(get(graph, optimizerNodes.gradients))
    #display(graph)

    println("errors before parallel computation:")
    resetGraph()
    @show lossParallel = nodeValue(graph, training.loss)
    gradientParallel = nodeValue(graph, optimizerNodes.gradients)
    #display(graph)

    @show norm(lossParallel .- lossSerial)
    @show norm(values(gradientParallel) .- gradientSerial)

    @test norm(lossParallel .- lossSerial) > 1e-8
    @test norm(values(gradientParallel) .- gradientSerial) > 1e-8

    #display(graph)

    ### serial computation
    println(@bold @red "Parallel computation: ")
    resetGraph()
    resetLog!(graph)
    computeSpawn!(graph)
    @show Int(maximum(graph.count))
    @test maximum(graph.count) == 0

    request(graph, training.loss)
    request(graph, optimizerNodes.gradients)
    wait(graph, training.loss)
    wait(graph, optimizerNodes.gradients)
    @show Int(maximum(graph.count))
    @test maximum(graph.count) == 1

    println("errors after parallel computation:")
    @show lossParallel

    @show norm(lossParallel .- lossSerial)
    @show norm(values(gradientParallel) .- gradientSerial)

    @test norm(lossParallel .- lossSerial) < 1e-8
    @test norm(values(gradientParallel) .- gradientSerial) < 1e-8

    #@show graph.enableTask
    #display(graph.tasks)
    computeUnspawn!(graph)
    #@show graph.enableTask
    #display(graph.tasks)
end

################################
## Checking timing & allocations
################################

@testset "NN: check timing and allocations" begin
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10
    @show nSamples = Int(round(10e9 / (sum(length.(W)) * trainingBatchSize)))

    lossType = :sse # TODO `loss_grad_Enzyme` given error with :mse
    (graph, theta, inference, training, optimizerNodes...) = denseChain(TypeValue;
        nNodes,
        inferenceBatchSize=1,
        trainingBatchSize,
        loss=lossType, optimizer=:gradDescent)

    """set all values to zero and reinitialize variables"""
    function resetGraph()
        for node in graph.nodes
            if !ComputationGraphs.noComputation(node)
                val = nodeValue(node)
                fill!(val, 0.0)
            end
        end
        set!(graph, values(theta), (W..., b...))
        set!(graph, inference.input, input)
        set!(graph, training.input, batchInput)
        set!(graph, training.reference, batchReference)
        set!(graph, optimizerNodes.eta, 1e-3)
        syncValid(graph)  # TODO: needs to be automated 
    end
    resetGraph()

    println(@bold @green "Serial computation: set(batch), compute!(loss+grad) (for $(trainingBatchSize) batch): ")

    BLAS.set_num_threads(6)
    @show Threads.nthreads()
    @show BLAS.get_num_threads()
    @show Base.JLOptions().opt_level

    resetLog!(graph)
    bmkSerial = @benchmark begin
        compute!($graph, $training.loss)
        compute!($graph, $optimizerNodes.gradients)
    end evals = 1 samples = nSamples setup = (
        set!($graph, $training.input, $batchInput))

    display(bmkSerial)
    @show Int(maximum(graph.count))
    @test bmkSerial.allocs == 0
    @test length(bmkSerial.times) == nSamples     # make sure there was time to run all samples
    @test all(graph.count .<= nSamples + 1) # each node computed, at most, once

    println(@bold @red "Parallel computation: set(batch), compute!(loss+grad) (for $(trainingBatchSize) batch): ")


    BLAS.set_num_threads(3)
    @show Threads.nthreads()
    @show BLAS.get_num_threads()
    @show Base.JLOptions().opt_level

    computeSpawn!(graph)
    resetLog!(graph)
    bmkParallel = @benchmark begin
        request($graph, $training.loss)
        request($graph, $optimizerNodes.gradients)
        wait($graph, $training.loss)
        wait($graph, $optimizerNodes.gradients)
    end evals = 1 samples = nSamples setup = (
        set!($graph, $training.input, $batchInput),
        syncValid($graph)) # TODO: needs to be automated 

    display(bmkParallel)
    @show Int(maximum(graph.count))
    @test bmkParallel.allocs == 0
    @test length(bmkParallel.times) == nSamples # make sure there was time to run all samples
    @test all(graph.count .<= nSamples + 1) # each node computed, at most, once

    computeUnspawn!(graph)

    timeSerial = median(bmkSerial.times)
    timeParallel = median(bmkParallel.times)
    println(@bold @red @sprintf("Improvement of parallel over serial = %7.3f us (%.1f %%)\n",
        1e-3 * timeSerial - 1e-3timeParallel, 100 * (timeSerial - timeParallel) / timeParallel))

    BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5
end

#= BLAS 1, -O2   2025/9/3
Serial computation: set(batch), compute!(loss+grad) (for 2000 batch): 
Threads.nthreads() = 6
BLAS.get_num_threads() = 6
(Base.JLOptions()).opt_level = 2
BenchmarkTools.Trial: 227 samples with 1 evaluation per sample.
 Range (min … max):  23.773 ms … 35.868 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     26.481 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   26.844 ms ±  1.847 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

          ▃ ▃█▅ ▄ ▅▂▄▆▆▃▁  ▂                                   
  ▃▃▃▅▄▄▄██▇███▆█▅███████▅▇█▇▅▄▆▅▅▆▄▄▃▄▄▅▃▁▃▃▅▁▁▃▅▃▃▄▄▁▄▁▃▃▁▃ ▄
  23.8 ms         Histogram: frequency by time        32.2 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
Int(maximum(graph.count)) = 228
Parallel computation: set(batch), compute!(loss+grad) (for 2000 batch): 
Threads.nthreads() = 6
BLAS.get_num_threads() = 3
(Base.JLOptions()).opt_level = 2
Unspawning computation nodes: done
Spawning computation nodes  : 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 31 32 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 50 51 52 53 done
BenchmarkTools.Trial: 227 samples with 1 evaluation per sample.
 Range (min … max):  20.020 ms … 26.644 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     21.911 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   22.154 ms ±  1.306 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

        ▂  ▁▂▄   ▅▃▁▅  █ ▃▁ ▁                                  
  ▄▃▄▅█▇█▇▆███▆▆▇████▆██▅████▆▄▆▃▄▅▅▆▅▄▃▃▄▁▅▄▄▄▃▆▁▃▆▄▃▃▁▁▃▁▁▃ ▄
  20 ms           Histogram: frequency by time        25.8 ms <

 Memory estimate: 4.91 KiB, allocs estimate: 30.
Int(maximum(graph.count)) = 228
Unspawning computation nodes: 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 31 32 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 50 51 52 53 done
Improvement of parallel over serial = 4570.257 us (20.9 %)
=#
