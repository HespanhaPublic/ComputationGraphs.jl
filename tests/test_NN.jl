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
    BLAS.set_num_threads(3)
    BLAS.set_num_threads(1) # for easier comparison
    @show Threads.nthreads()
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
    desiredOutput = randn(TypeValue, nNodes[end])
    # batch
    trainingBatchSize = 1000
    batchInput = randn(TypeValue, nNodes[1], trainingBatchSize)
    batchReference = randn(TypeValue, nNodes[end], trainingBatchSize)
    # error tolerance for testing
    tol = 1e-8
end

@testset "NN: check dependencies" begin
    (; graph, theta, inference, training) = denseChain(TypeValue;
        nNodes,
        inferenceBatchSize=1,
        trainingBatchSize=0, # no loss and no gradients
        loss=:mse, optimizer=:none)

    set!(graph, values(theta), (W..., b...))
    set!(graph, inference.input, input) # changing input invalidates all

    display(graph)
    println("output:")
    display(graph, inference.output)

    #@code_warntype compute!(graph,inference.output)
    #@code_warntype compute!(graph,training.loss)

    @test all(isa(graph.nodes[i], ComputationGraphs.AbstractConstantNode) ||
              isa(graph.nodes[i], ComputationGraphs.NodeVariable) ||
              graph.validValue[i] == false for i in eachindex(graph.nodes)) # all invalid
    #display(graph)
    compute!(graph; force=true)
    @test all(graph.validValue[i] for i in eachindex(graph.nodes)) # all valid
    #display(graph)

    set!(graph, inference.input, input) # changing input invalidates all
    #display(graph)
    if length(size(input)) == 1
        @test all(isa(graph.nodes[i], ComputationGraphs.AbstractConstantNode) ||
                  isa(graph.nodes[i], ComputationGraphs.NodeVariable) ||
                  graph.validValue[i] == false for i in eachindex(graph.nodes)) # all invalid
    end

    compute!(graph; force=true)
    @test all(graph.validValue[i] for i in eachindex(graph.nodes)) # all valid
    #display(graph)

    # changing W[end=3] only invalidates last layer (but most of the gradient gets messed up)
    set!(graph, theta.layer2_W, W[end-1])
    #display(graph)
    # check location if Affine node
    @test all(isa.(graph.nodes[10:3:end], Ref(ComputationGraphs.NodeAffine)))
    @test all(graph.validValue[1:9] .== true)
    if length(size(input)) == 1
        @test all(graph.validValue[10:12] .== false) # affine, subtract, norm2
    end
    #display(graph)
    get(graph, inference.output) # recompute output needs to recompute all
    #display(graph)
    @test all(graph.validValue .== true)

    #display(graph)
end

@testset "NN: check accuracy" begin
    lossType = :sse # TODO `loss_grad_Enzyme` given error with :mse
    (graph, theta, inference, training, optimizerNodes...) = denseChain(TypeValue;
        nNodes,
        inferenceBatchSize=1,
        trainingBatchSize,
        loss=lossType, optimizer=:gradDescent)

    set!(graph, values(theta), (W..., b...))
    set!(graph, inference.input, input)
    set!(graph, training.input, batchInput)
    set!(graph, training.reference, batchReference)
    set!(graph, optimizerNodes.eta, 1e-3)

    display(graph)
    println("output:")
    display(graph, inference.output)
    println("loss:")
    display(graph, training.loss)
    println("grad W[3]:")
    display(graph, optimizerNodes.gradients.layer3_W)
    println("grad W[1]:")
    display(graph, optimizerNodes.gradients.layer1_W)
    println("grad b[1]:")
    display(graph, optimizerNodes.gradients.layer1_b)
    println("loss+grad W[3]:")
    display(graph, training.loss, optimizerNodes.gradients.layer3_W)
    println("loss+grad W[1]:")
    display(graph, training.loss, optimizerNodes.gradients.layer1_W)
    println("loss+grad b[1]:")
    display(graph, training.loss, optimizerNodes.gradients.layer1_b)


    compute!(graph; force=true)
    #display(graph)

    (model_Zygote, loss_Zygote, loss_grad_Zygote,) =
        denseChain_FluxZygote(TypeValue; W, b, loss=lossType)
    (model_Enzyme, loss_Enzyme, loss_grad_Enzyme,) =
        denseChain_FluxEnzyme(TypeValue; W, b, loss=lossType)

    # check losses
    println("Checking loss once")
    @show loss = get(graph, training.loss)
    @show loss_Z = loss_Zygote(model_Zygote, batchInput, batchReference)
    @show loss_E = loss_Enzyme(model_Enzyme, batchInput, batchReference)

    @test norm(loss .- loss_Z) / length(batchReference) < tol
    @test norm(loss .- loss_E) / length(batchReference) < tol

    # check loss & grad
    println("Computing loss & grad using Zygote")
    (loss_Z, grad_Z) = loss_grad_Zygote(model_Zygote, batchInput, batchReference)
    println("Computing loss & grad using Enzyme")
    # TODO `loss_grad_Enzyme` given error with :mse
    (loss_E, grad_E) = loss_grad_Enzyme(model_Enzyme, batchInput, batchReference)

    println("Checking loss again")
    @test norm(loss .- loss_Z) / length(batchReference) < tol
    @test norm(loss .- loss_E) / length(batchReference) < tol

    for l in 1:length(nNodes)-1
        @printf("Checking layer %d:", l)
        layer_zygote = grad_Z[1].layers[l]
        layer_enzyme = grad_E[1].layers[l]

        dloss_dW = get(graph, optimizerNodes.gradients[Symbol("layer", l, "_W")])
        dloss_db = get(graph, optimizerNodes.gradients[Symbol("layer", l, "_b")])

        @test norm(layer_zygote.weight - dloss_dW) / length(dloss_dW) < tol
        @test norm(layer_zygote.bias - dloss_db) / length(dloss_db) < tol

        @test norm(layer_enzyme.weight - dloss_dW) / length(dloss_dW) < tol
        @test norm(layer_enzyme.bias - dloss_db) / length(dloss_db) < tol
        println(" no error")
    end

end

################################
## Checking timing & allocations
################################
@testset "NN: check timing and allocations without gradient" begin
    @show Threads.nthreads()
    @show BLAS.get_num_threads()
    @show Base.JLOptions().opt_level

    lossType = :sse # TODO `loss_grad_Enzyme` given error with :mse
    (graph, theta, inference, training, optimizerNodes...) = denseChain(TypeValue;
        nNodes,
        inferenceBatchSize=1,
        trainingBatchSize,
        loss=lossType, optimizer=:none)
    @test length(graph.nodes) == 22

    set!(graph, values(theta), (W..., b...))
    set!(graph, inference.input, input)
    set!(graph, training.input, batchInput)
    set!(graph, training.reference, batchReference)

    @time compute!(graph; force=true)

    function test_(graph)
        for (id, node) in enumerate(graph.nodes)
            display(node)
            if !isa(node, ComputationGraphs.NodeVariable)
                display(node)
                @time ComputationGraphs.compute_node!(node)
                #@code_warntype compute!(graph, graph.nodes[18])
                #graph.validValue[i] = false
                #@time compute!(graph, n)
            end
        end
    end
    #test_(graph)

    resetLog!(graph)
    @time compute!(graph, force=true)
    #@show graph.count

    @time output = get(graph, inference.output)

    ## allocs
    nSamples = 3000

    print("set!(W[1]]): ")
    resetLog!(graph)
    bmk = @benchmark begin
        set!($graph, $theta.layer1_W, $W[1])
    end evals = 1 samples = nSamples
    display(bmk)
    #@show graph.count
    @test bmk.allocs == 0
    @test all(graph.count .<= nSamples + 1) # each node computed, at most, once

    print("set!(Ws,bs): ")
    resetLog!(graph)
    thetaNodes = values(theta)
    thetaVals = (W..., b...)
    bmk = @benchmark begin
        set!($graph, $thetaNodes, $thetaVals)
    end evals = 1 samples = nSamples
    display(bmk)
    #@show graph.count
    @test bmk.allocs == 0
    @test all(graph.count .<= nSamples + 1) # each node computed, at most, once

    println(@bold @green "compute!(graph,force=true) (graph => output for input & $(trainingBatchSize) batch): ")

    resetLog!(graph)
    bmk = @benchmark compute!($graph, force=true) evals = 1 samples = nSamples
    display(bmk)
    #@show graph.count
    @test bmk.allocs == 0
    @test all(graph.count .<= nSamples + 1) # each node computed, at most, once

    print(@red @bold @sprintf("   %.2f GFlops\n",
        ((trainingBatchSize + 1) * sum(inference.nOps)       # compute outputs
         +
         trainingBatchSize * (3 * length(input)))  # compute losses 
        /
        median(bmk.times)))
end

#= BLAS 1, -O2   2025/9/3
Threads.nthreads() = 6
BLAS.get_num_threads() = 1
(Base.JLOptions()).opt_level = 2
  0.000619 seconds (13 allocations: 544 bytes)
  0.000584 seconds (13 allocations: 544 bytes)
  0.000010 seconds (10 allocations: 528 bytes)
set!(W[1]): BenchmarkTools.Trial: 10000 samples with 993 evaluations per sample.
 Range (min … max):  25.102 ns … 231.269 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     26.307 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   29.258 ns ±   6.056 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █▆▃▃▃▃▁▂▁▂▁▂▁▆▃▄▁▂▂ ▆▃ ▄▁ ▂     ▁       ▁                    ▂
  ███████████████████▅███████▅▅█▆▁█▇▅▅█▆▅▄█▆▅▅▅▆▄▄▆▅▄▅▄▄▄▄▅▃▆▄ █
  25.1 ns       Histogram: log(frequency) by time      50.2 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
compute!(graph,force=true) (graph => output for input & 1000 batch): 
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):  305.084 μs …  1.298 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     314.171 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   343.419 μs ± 59.548 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █▇▆▂▂      ▆▃▄▁▁    ▂▁    ▁                                  ▂
  ███████▇█▇███████▇████████████▇▇▇█▇▇██▇▇██▇▇▇▇▆▆▆▆▇▆▅▆▅▅▆▅▄▅ █
  305 μs        Histogram: log(frequency) by time       573 μs <

 Memory estimate: 0 bytes, allocs estimate: 0.
   23.39 GFlops
=#

@testset "NN: check timing and allocations with gradient" begin
    @show Threads.nthreads()
    @show BLAS.get_num_threads()
    @show Base.JLOptions().opt_level

    lossType = :sse # TODO `loss_grad_Enzyme` given error with :mse
    (graph, theta, inference, training, optimizerNodes...) = denseChain(TypeValue;
        nNodes,
        inferenceBatchSize=1,
        trainingBatchSize,
        loss=lossType, optimizer=:gradDescent)

    if length(size(input)) == 1
        @test length(graph.nodes) == 56
    end

    set!(graph, values(theta), (W..., b...))
    set!(graph, inference.input, input)
    set!(graph, training.input, batchInput)
    set!(graph, training.reference, batchReference)
    set!(graph, optimizerNodes.eta, 1e-3)

    @time compute!(graph; force=true)

    function test_(graph)
        for (i, n) in enumerate(graph.nodes)
            @show (i, typeof(n))
            @time "   " compute_node!(n)
            # @code_warntype compute!(graph, graph.nodes[18])
            #graph.validValue[i] = false
            #@time compute!(graph, n)
        end
    end
    #test_(graph)
    #@code_warntype compute!(graph, training.loss)
    #@code_warntype ComputationGraphs.computeParents!(graph, training.loss.parentIds)
    #@code_warntype compute!(graph)
    #@code_warntype graph.computeFunctions[training.loss.id](graph)
    #@code_warntype training.loss.compute!(graph)

    @time output = get(graph, inference.output)
    @show loss = get(graph, training.loss)

    ## allocs

    nSamples = 3000

    println(@bold @green "compute!(all,force=false) (graph => loss+grad for input & $(trainingBatchSize) batch): ")
    resetLog!(graph)
    bmk = @benchmark compute!($graph, force=false) evals = 1 samples = nSamples
    display(bmk)
    #@show graph.count
    @test bmk.allocs == 0
    @test all(graph.count .<= 0) # all nodes have been previously computed

    println(@bold @green "compute!(all,force=true) (graph => loss+grad for input & $(trainingBatchSize) batch): ")
    resetLog!(graph)
    bmk = @benchmark compute!($graph, force=true) evals = 1 samples = nSamples
    display(bmk)
    #@show graph.count
    @test bmk.allocs == 0
    @test all(graph.count .<= nSamples + 1) # each node computed, at most, once

    println(@bold @green "set(batch), compute!(loss+grad) (for $(trainingBatchSize) batch): ")
    what = optimizerNodes.gradients
    #display(what)
    resetLog!(graph)
    bmk = @benchmark begin
        compute!($graph, $training.loss)
        compute!($graph, $what)
    end evals = 1 samples = nSamples setup = (
        set!($graph, $training.input, $batchInput))
    display(bmk)
    #@show graph.count
    @test bmk.allocs == 0
    @test all(graph.count .<= nSamples + 1) # each node computed, at most, once

end

## THE SECOND SET OF TIMES LOOKS VERY GOOD: PROBABLY OBTAINED PRIOR TO USING CLOSURE ??? (NOTE ALLOCATIONS)
#= BLAS 1, -O2
  0.001674 seconds (13 allocations: 544 bytes)
  0.000005 seconds (8 allocations: 384 bytes)
loss = get(graph, training.loss) = fill(4.615603250687031e8)
compute!(all,force) (graph => loss+grad for input & 1000 batch): 
BenchmarkTools.Trial: 5000 samples with 1 evaluation per sample.
 Range (min … max):  654.083 μs … 84.016 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     948.742 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   988.337 μs ±  1.245 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █▆▁         ▃▂▁▂▁▁▁                                           
  ███▇▆▄▅▄▄▄▅▇███████▇▇▆▆▅▅▅▄▄▄▃▄▄▄▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂ ▄
  654 μs          Histogram: frequency by time         1.86 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
set(batch), compute!(loss+grad) (for 1000 batch): 
BenchmarkTools.Trial: 7067 samples with 1 evaluation per sample.
 Range (min … max):  559.535 μs …   2.035 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     629.621 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   700.367 μs ± 152.652 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

    █▇▄▁                                                         
  ▃▇████▆▆▆▄▃▃▄▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  560 μs           Histogram: frequency by time          1.2 ms <

 Memory estimate: 1.02 KiB, allocs estimate: 26.
=#

#= 2025/9/3
Threads.nthreads() = 6
BLAS.get_num_threads() = 1
(Base.JLOptions()).opt_level = 2
  0.001530 seconds (13 allocations: 544 bytes)
  0.000010 seconds (10 allocations: 528 bytes)
loss = get(graph, training.loss) = fill(4.615603250687031e8)
compute!(all,force=false) (graph => loss+grad for input & 1000 batch): 
BenchmarkTools.Trial: 10000 samples with 990 evaluations per sample.
 Range (min … max):  32.010 ns … 160.425 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     32.808 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   35.113 ns ±   6.603 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █▆▆▁▁                ▂  ▃  ▂▃▁ ▄▃▁▁▂                         ▂
  ██████▇▅▆▅▇▇▄█▅▇▄▅▇▄▁█▃▃██▆███▇███████▆▄█▅▅▄▅▅▆▅▅▄▅▄▁▄▄▃▄▃▁▄ █
  32 ns         Histogram: log(frequency) by time      56.7 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
compute!(all,force=true) (graph => loss+grad for input & 1000 batch): 
BenchmarkTools.Trial: 5281 samples with 1 evaluation per sample.
 Range (min … max):  827.707 μs …   1.832 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     911.738 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   941.380 μs ± 123.375 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▄█        ▂                                                    
  ██▇▄▃▃▃▃▃▂█▅▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  828 μs           Histogram: frequency by time         1.32 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
set(batch), compute!(loss+grad) (for 1000 batch): 
BenchmarkTools.Trial: 5422 samples with 1 evaluation per sample.
 Range (min … max):  723.691 μs …   1.959 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     903.824 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   904.466 μs ± 152.869 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █▃                                                             
  ██▆▄▃▃▄▄▃▂▃▃▃▃▃▃▃▂▂▂▃▆▅▄▄▄▃▄▃▅▄▄▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  724 μs           Histogram: frequency by time         1.32 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
=#

@testset "NN:compare flux times" begin
    @show Threads.nthreads()
    @show BLAS.get_num_threads()
    @show Base.JLOptions().opt_level

    lossSymbol = :sse
    (model_Zygote, loss_Zygote, loss_grad_Zygote,) =
        denseChain_FluxZygote(TypeValue; W, b, loss=lossSymbol)
    (model_Enzyme, loss_Enzyme, loss_grad_Enzyme,) =
        denseChain_FluxEnzyme(TypeValue; W, b, loss=lossSymbol)

    println(@bold @blue "Flux+Zygote loss for input & $(trainingBatchSize) batch")
    bmk = @benchmark begin
        $loss_Zygote($model_Zygote, $input, $desiredOutput)
        $loss_Zygote($model_Zygote, $batchInput, $batchReference)
    end
    display(bmk)

    println(@bold @blue "Flux+Zygote loss+grad for input & $(trainingBatchSize) batch")
    bmk = @benchmark begin
        $loss_Zygote($model_Zygote, $input, $desiredOutput)
        $loss_grad_Zygote($model_Zygote, $batchInput, $batchReference)
    end
    display(bmk)

    println(@bold @cyan "Flux+Enzyme loss for input & $(trainingBatchSize) batch ")
    bmk = @benchmark begin
        $loss_Enzyme($model_Enzyme, $input, $desiredOutput)
        $loss_Enzyme($model_Enzyme, $batchInput, $batchReference)
    end
    display(bmk)

    println(@bold @cyan "Flux+Enzyme loss+grad for input & $(trainingBatchSize) batch")
    bmk = @benchmark begin
        $loss_Enzyme($model_Enzyme, $input, $desiredOutput)
        $loss_grad_Enzyme($model_Enzyme, $batchInput, $batchReference)
    end
    display(bmk)
end

#= 2025/9/3
Threads.nthreads() = 6
BLAS.get_num_threads() = 1
(Base.JLOptions()).opt_level = 2
denseChain_FluxZygote:
Chain(
  Dense(30 => 20, relu),                # 620 parameters
  Dense(20 => 50, relu),                # 1_050 parameters
  Dense(50 => 40),                      # 2_040 parameters
)                   # Total: 6 arrays, 3_710 parameters, 29.289 KiB.
loss = :sse
denseChain_FluxZygote:
Chain(
  Dense(30 => 20, relu),                # 620 parameters
  Dense(20 => 50, relu),                # 1_050 parameters
  Dense(50 => 40),                      # 2_040 parameters
)                   # Total: 6 arrays, 3_710 parameters, 29.289 KiB.
loss = :sse
Flux+Zygote loss for input & 1000 batch
BenchmarkTools.Trial: 6634 samples with 1 evaluation per sample.
 Range (min … max):  359.255 μs … 51.697 ms  ┊ GC (min … max):  0.00% … 98.25%
 Time  (median):     454.524 μs              ┊ GC (median):     0.00%
 Time  (mean ± σ):   750.791 μs ±  1.937 ms  ┊ GC (mean ± σ):  38.18% ± 16.82%

  █▃  ▁                                                        ▁
  ██▆▆█▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▁▁▃▄▃▃▅▆▆▅▅▅▄▅▃▃▄▆▆▅▄▆▆▄▅▆▄▅▄▅▃▄▃▅▄ █
  359 μs        Histogram: log(frequency) by time      8.94 ms <

 Memory estimate: 1.19 MiB, allocs estimate: 20.
Flux+Zygote loss+grad for input & 1000 batch
BenchmarkTools.Trial: 2163 samples with 1 evaluation per sample.
 Range (min … max):  1.089 ms … 83.556 ms  ┊ GC (min … max):  0.00% … 98.22%
 Time  (median):     1.312 ms              ┊ GC (median):     0.00%
 Time  (mean ± σ):   2.305 ms ±  4.769 ms  ┊ GC (mean ± σ):  42.09% ± 23.53%

  █▅                                                          
  ██▆▃▄█▁▁▁▃▃▄▅▅▆▇▆▆▆▆▆▆▁▅▄▅▃▅▅▅▅▅▅▁▄▃▅▅▅▁▃▅▅▃▄▃▃▃▃▅▅▃▁▄▁▄▁▃ █
  1.09 ms      Histogram: log(frequency) by time     18.9 ms <

 Memory estimate: 2.98 MiB, allocs estimate: 135.
Flux+Enzyme loss for input & 1000 batch 
BenchmarkTools.Trial: 5597 samples with 1 evaluation per sample.
 Range (min … max):  370.751 μs … 133.985 ms  ┊ GC (min … max):  0.00% … 99.59%
 Time  (median):     458.735 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):   889.000 μs ±   3.571 ms  ┊ GC (mean ± σ):  47.68% ± 16.76%

  █▂                                                            ▁
  ██▃▆▄▁▁▁▁▁▁▁▁▁▁▁▁▃▁▃▁▁▄▄▄▅▅▆▅▅▅▄▄▃▅▅▅▆▄▅▅▄▃▄▄▃▄▃▅▄▄▃▁▁▄▄▄▄▅▅▄ █
  371 μs        Histogram: log(frequency) by time       11.2 ms <

 Memory estimate: 1.19 MiB, allocs estimate: 20.
Flux+Enzyme loss+grad for input & 1000 batch
BenchmarkTools.Trial: 876 samples with 1 evaluation per sample.
 Range (min … max):  3.095 ms … 161.294 ms  ┊ GC (min … max):  0.00% … 93.61%
 Time  (median):     4.293 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   5.747 ms ±   8.123 ms  ┊ GC (mean ± σ):  23.81% ± 19.04%

   ▂█▄                                                         
  ▇████▆▄▆▆▅▄▄▅▁▆▄▅▄▄▅▅▆▄▄▄▄▄▅▁▅▄▄▄▄▄▄▄▁▄▄▁▁▄▁▄▄▄▁▄▁▁▁▁▁▁▁▄▁▄ ▇
  3.1 ms       Histogram: log(frequency) by time      29.7 ms <

 Memory estimate: 2.45 MiB, allocs estimate: 316.
=#