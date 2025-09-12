#! julia --threads=6 --project=. tests/test_CodeGeneration_module.jl

using Revise

using LinearAlgebra
using Random
using BenchmarkTools
using Printf
using Term

using ComputationGraphs

using Test

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5

begin
    BLAS.set_num_threads(5)
    @show Threads.nthreads() # does not seem to make a difference 
    @show BLAS.get_num_threads()
    @show Base.JLOptions().opt_level
end

begin
    # NN Parameters
    TypeValue = Float64
    nNodes = [3, 2, 5, 4] .* 20
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
    # options
    unrolled = false
    # BLAS 5, -O2, unrolled=false (2nd time): median 2.72μs, 576μs, 1.76ms

    nSamples = convert(Int, round(100 * 20000 / trainingBatchSize))
end

@testset "CodeGeneration: Code+set!+get!+save" begin
    lossType = :sse # TODO `loss_grad_Enzyme` given error with :mse
    (graph, theta, inference, training, optimizerNodes...) = denseChain(TypeValue;
        nNodes,
        inferenceBatchSize=1,
        trainingBatchSize,
        loss=lossType, optimizer=:gradDescent)

    ## Check constructor
    code = Code(graph, "tmp_mod";
        type=:module, unrolled)
    #display(code)
    @test isempty(code.inits)
    @test isempty(code.sets)
    @test isempty(code.gets)

    ## Check Sets
    sets!(code,
        inference.input => "setInferenceInput",
        training.input => "setTrainingInput",
        training.reference => "setTrainingReference",
    )
    #display(code)
    #print("# Sets\n", code.sets)
    @test isempty(code.inits)
    @test !isempty(code.sets)
    @test isempty(code.gets)

    sets!(code, optimizerNodes.eta => "setEta")
    #display(code)
    @test isempty(code.inits)
    @test !isempty(code.sets)
    @test isempty(code.gets)

    sets!(code, theta => "setTheta")
    #display(code)
    @test isempty(code.inits)
    @test !isempty(code.sets)
    @test isempty(code.gets)

    ## Check Gets
    gets!(code,
        inference.output => "getOutput",
        training.loss => "getLoss",
    )
    #display(code)
    #print("# Gets\n", code.gets)
    @test isempty(code.inits)
    @test !isempty(code.sets)
    @test !isempty(code.gets)

    gets!(code,
        optimizerNodes.gradients => "get_gradient",
    )
    #display(code)
    @test isempty(code.inits)
    @test !isempty(code.sets)
    @test !isempty(code.gets)

    ## Check save
    save("tests/tmp_mod_code.jl", code)

    @test !isempty(code.inits)
end

@testset "CodeGeneration: execution" begin
    lossType = :sse # TODO `loss_grad_Enzyme` given error with :mse
    (graph, theta, inference, training, optimizerNodes...) = denseChain(TypeValue;
        nNodes,
        inferenceBatchSize=1,
        trainingBatchSize,
        loss=lossType, optimizer=:gradDescent)


    ## Check code usage
    includet("tmp_mod_code.jl")
    using .tmp_mod
    revise(tmp_mod; force=true)

    # check sets
    tmp_mod.setTheta(W..., b...)
    set!(graph, values(theta), (W..., b...))
    @test theta.layer1_W.id == 1
    @test all(tmp_mod.Node1 .== W[1])
    @test all(tmp_mod.Node2 .== W[2])
    @test all(tmp_mod.Node3 .== W[3])

    @test theta.layer1_b.id == 4
    @test all(tmp_mod.Node4 .== b[1])
    @test all(tmp_mod.Node5 .== b[2])
    @test all(tmp_mod.Node6 .== b[3])

    tmp_mod.setInferenceInput(input)
    set!(graph, inference.input, input)
    @test inference.input.id == 7
    @test all(tmp_mod.Node7 .== input)

    tmp_mod.setTrainingInput(batchInput)
    set!(graph, training.input, batchInput)
    @test training.input.id == 8
    @test all(tmp_mod.Node8 .== batchInput)

    tmp_mod.setTrainingReference(batchReference)
    set!(graph, training.reference, batchReference)
    @test training.reference.id == 9
    @test all(tmp_mod.Node9 .== batchReference)

    tmp_mod.setEta(eta)
    set!(graph, optimizerNodes.eta, eta)
    @test optimizerNodes.eta.id == 23
    @test all(tmp_mod.Node23 .== eta)

    #@code_warntype tmp_mod.setInferenceInput(input)

    ## check correctness

    # values from module
    @show loss = tmp_mod.getLoss()
    gradient = tmp_mod.get_gradient()

    # value from graph
    @show loss1 = get(graph, training.loss)
    @test norm(loss .- loss1) < 1e-8

    for (i, gradNode) in enumerate(optimizerNodes.gradients)
        # value from graph
        gradValue = get(graph, gradNode)
        @test norm(gradient[i] .- gradValue) < 1e-8
    end

    ## Check timing --original code

    print(@bold @blue "set(input)+get(output): ")
    bmk_input_output = @benchmark begin
        set!($graph, input_, $input)
        output = get($graph, output_)
    end evals = 1 samples = nSamples setup = (
        input_ = $inference.input; output_ = $inference.output)
    display(bmk_input_output)
    @test bmk_input_output.allocs == 0
    println(@blue @bold @sprintf("\n# ops/sec = %.3f G\n", sum(inference.nOps) / mean(bmk_input_output.times)))

    print(@bold @green "set(batchInput)+get(loss): ")
    bmk_input_loss = @benchmark begin
        set!($graph, batchInput_, $batchInput)
        loss = get($graph, loss_)
    end evals = 1 samples = nSamples setup = (
        batchInput_ = $training.input;
        loss_ = $training.loss;
        gradient_ = $optimizerNodes.gradients)
    display(bmk_input_loss)
    @test bmk_input_loss.allocs == 0

    print(@bold @green "set(batchInput)+get(loss,gradients): ")
    bmk_input_loss_grad = @benchmark begin
        set!($graph, batchInput_, $batchInput)
        loss = get($graph, loss_)
        # gradient = get($graph, gradient_) # not allocation free
        compute!($graph, gradient_) # not allocation free
    end evals = 1 samples = nSamples setup = (
        batchInput_ = $training.input;
        loss_ = $training.loss;
        gradient_ = $optimizerNodes.gradients)
    display(bmk_input_loss_grad)
    @test bmk_input_loss_grad.allocs == 0


    ## Check allocations -- generated code
    #@code_warntype getLoss(tmp_struct)
    print(@bold @green "set(input)+get(output): ")
    bmk_CG_input_output = @benchmark begin
        tmp_mod.setInferenceInput($input)
        output = tmp_mod.getOutput()
    end evals = 1 samples = nSamples
    display(bmk_CG_input_output)
    @test bmk_CG_input_output.allocs == 0
    println(@red @bold @sprintf("\n# ops/sec = %.3f G\n", sum(inference.nOps) / mean(bmk_CG_input_output.times)))

    print(@bold @green "set(batchInput)+get(loss): ")
    bmk_CG_input_loss = @benchmark begin
        tmp_mod.setTrainingInput($batchInput)
        loss = tmp_mod.getLoss()
    end evals = 1 samples = nSamples
    display(bmk_CG_input_loss)
    @test bmk_CG_input_loss.allocs == 0

    print(@bold @green "set(batchInput)+get(loss,gradients): ")
    bmk_CG_input_loss_grad = @benchmark begin
        tmp_mod.setTrainingInput($batchInput)
        loss = tmp_mod.getLoss()
        gradient = tmp_mod.get_gradient()
    end evals = 1 samples = nSamples
    display(bmk_CG_input_loss_grad)
    @test bmk_CG_input_loss_grad.allocs == 0

    println(@red @bold "Code generation time gains:")
    println(@red @bold @sprintf("    set(input)+get(output):          %8.3f us (%9.3f -> %9.3f)  (<0 means code gen slower)",
        1e-3median(bmk_input_output.times) - 1e-3median(bmk_CG_input_output.times),
        1e-3median(bmk_input_output.times), 1e-3median(bmk_CG_input_output.times)
    ))
    println(@red @bold @sprintf("    set(batchInput)+get(loss):       %8.3f us (%9.3f -> %9.3f)  (<0 means code gen slower)",
        1e-3median(bmk_input_loss.times) - 1e-3median(bmk_CG_input_loss.times),
        1e-3median(bmk_input_loss.times), 1e-3median(bmk_CG_input_loss.times)
    ))
    println(@red @bold @sprintf("    set(batchInput)+get(loss,grads): %8.3f us (%9.3f -> %9.3f)  (<0 means code gen slower)",
        1e-3median(bmk_input_loss_grad.times) - 1e-3median(bmk_CG_input_loss_grad.times),
        1e-3median(bmk_input_loss_grad.times), 1e-3median(bmk_CG_input_loss_grad.times)
    ))
    #rm("tests/tmp_mod_code.jl")
end
