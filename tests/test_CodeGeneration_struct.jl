#! julia --threads=6 --project=. tests/test_CodeGeneration_struct.jl

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
    # BLAS 5, -O2, unrolled=false (2nd time): median 2.93μs, 614μs, 1.67ms

    eta = fill(1e-3)

    nSamples = convert(Int, round(100 * 15000 / trainingBatchSize))
end

@testset "CodeGeneration: Code+set!+get!+save" begin
    lossType = :sse # TODO `loss_grad_Enzyme` given error with :mse
    (graph, theta, inference, training, optimizerNodes...) = denseChain(TypeValue;
        nNodes,
        inferenceBatchSize=1,
        trainingBatchSize,
        loss=lossType, optimizer=:gradDescent)

    ## Check constructor
    code = Code(graph, "tmp_struct";
        type=:struct, unrolled,
        parallel=true)
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
    save("tests/tmp_struct_code.jl", code)

    @test !isempty(code.inits)
end

@testset "CodeGeneration: execution" begin
    lossType = :sse # TODO `loss_grad_Enzyme` given error with :mse
    (graph, theta, inference, training, optimizerNodes...) = denseChain(TypeValue;
        nNodes,
        inferenceBatchSize=1,
        trainingBatchSize,
        loss=lossType, optimizer=:gradDescent)

    Revise.track(Main, "tests/tmp_struct_code.jl"; mode=:eval) # includet default is not good enough
    revise() # seems needed after track
    tmp_struct = Tmp_struct()

    # check sets
    setTheta(tmp_struct, W..., b...)
    set!(graph, values(theta), (W..., b...))
    @test theta.layer1_W.id == 1
    @test all(tmp_struct.Node1 .== W[1])
    @test all(tmp_struct.Node2 .== W[2])
    @test all(tmp_struct.Node3 .== W[3])
    @test theta.layer1_b.id == 4
    @test all(tmp_struct.Node4 .== b[1])
    @test all(tmp_struct.Node5 .== b[2])
    @test all(tmp_struct.Node6 .== b[3])

    setInferenceInput(tmp_struct, input)
    set!(graph, inference.input, input)
    @test inference.input.id == 7
    @test all(tmp_struct.Node7 .== input)

    setTrainingInput(tmp_struct, batchInput)
    set!(graph, training.input, batchInput)
    @test training.input.id == 8
    @test all(tmp_struct.Node8 .== batchInput)

    setTrainingReference(tmp_struct, batchReference)
    set!(graph, training.reference, batchReference)
    @test training.reference.id == 9
    @test all(tmp_struct.Node9 .== batchReference)

    setEta(tmp_struct, eta)
    set!(graph, optimizerNodes.eta, eta)
    @test optimizerNodes.eta.id == 23
    @test all(tmp_struct.Node23 .== eta)

    #@code_warntype setInferenceInput(tmp_struct, input)

    ## check correctness

    # values from structure
    @show loss = getLoss(tmp_struct)
    gradient = get_gradient(tmp_struct)

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
        gradient = get($graph, gradient_)
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
        setInferenceInput(s, $input)
        output = getOutput(s)
    end evals = 1 samples = nSamples setup = (s = $tmp_struct)
    display(bmk_CG_input_output)
    @test bmk_CG_input_output.allocs == 0
    println(@red @bold @sprintf("\n# ops/sec = %.3f G\n", sum(inference.nOps) / mean(bmk_CG_input_output.times)))

    print(@bold @green "set(batchInput)+get(loss): ")
    bmk_CG_input_loss = @benchmark begin
        setTrainingInput(s, $batchInput)
        loss = getLoss(s)
    end evals = 1 samples = nSamples setup = (s = $tmp_struct)
    display(bmk_CG_input_loss)
    @test bmk_CG_input_loss.allocs == 0

    print(@bold @green "set(batchInput)+get(loss,gradients): ")
    bmk_CG_input_loss_grad = @benchmark begin
        setTrainingInput(s, $batchInput)
        loss = getLoss(s)
        gradient = get_gradient(s)
    end evals = 1 samples = nSamples setup = (s = $tmp_struct)
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
    #rm("tests/tmp_struct_code.jl")
end

begin
    BLAS.set_num_threads(2) # 2 seems good
    @show Threads.nthreads() # does not seem to make a difference 
    @show BLAS.get_num_threads()
    @show Base.JLOptions().opt_level
end

@testset "CodeGeneration: computeAll_parallel" begin
    lossType = :sse # TODO `loss_grad_Enzyme` given error with :mse
    (graph, theta, inference, training, optimizerNodes...) = denseChain(TypeValue;
        nNodes,
        inferenceBatchSize=1,
        trainingBatchSize,
        loss=lossType, optimizer=:gradDescent)

    Revise.track(Main, "tests/tmp_struct_code.jl"; mode=:eval) # includet default is not good enough
    revise() # seems needed after track
    tmp_struct = Tmp_struct()

    # check sets
    setTheta(tmp_struct, W..., b...)
    set!(graph, values(theta), (W..., b...))
    @test theta.layer1_W.id == 1
    @test all(tmp_struct.Node1 .== W[1])
    @test all(tmp_struct.Node2 .== W[2])
    @test all(tmp_struct.Node3 .== W[3])
    @test theta.layer1_b.id == 4
    @test all(tmp_struct.Node4 .== b[1])
    @test all(tmp_struct.Node5 .== b[2])
    @test all(tmp_struct.Node6 .== b[3])

    setInferenceInput(tmp_struct, input)
    set!(graph, inference.input, input)
    @test inference.input.id == 7
    @test all(tmp_struct.Node7 .== input)

    setTrainingInput(tmp_struct, batchInput)
    set!(graph, training.input, batchInput)
    @test training.input.id == 8
    @test all(tmp_struct.Node8 .== batchInput)

    setTrainingReference(tmp_struct, batchReference)
    set!(graph, training.reference, batchReference)
    @test training.reference.id == 9
    @test all(tmp_struct.Node9 .== batchReference)

    setEta(tmp_struct, eta)
    set!(graph, optimizerNodes.eta, eta)
    @test optimizerNodes.eta.id == 26
    @test all(tmp_struct.Node26 .== eta)

    #@code_warntype setInferenceInput(tmp_struct, input)

    computeAll_parallel(tmp_struct)

    ## check correctness
    @test training.loss.id == 25
    @show loss = tmp_struct.Node25
    @show loss1 = get(graph, training.loss)
    @test norm(loss .- loss1) < 1e-8

    @test optimizerNodes.gradients[1].id == 38
    @test optimizerNodes.gradients[2].id == 34
    @test optimizerNodes.gradients[3].id == 30
    @test optimizerNodes.gradients[4].id == 40
    @test optimizerNodes.gradients[5].id == 41
    @test optimizerNodes.gradients[6].id == 42
    @test norm(get(graph, optimizerNodes.gradients[1]) .- tmp_struct.Node38) < 1e-8
    @test norm(get(graph, optimizerNodes.gradients[2]) .- tmp_struct.Node34) < 1e-8
    @test norm(get(graph, optimizerNodes.gradients[3]) .- tmp_struct.Node30) < 1e-8
    @test norm(get(graph, optimizerNodes.gradients[4]) .- tmp_struct.Node40) < 1e-8
    @test norm(get(graph, optimizerNodes.gradients[5]) .- tmp_struct.Node41) < 1e-8
    @test norm(get(graph, optimizerNodes.gradients[6]) .- tmp_struct.Node42) < 1e-8

    print(@bold @green "set(batchInput)+computeAll_parallel(): ")
    bmk_CG_all = @benchmark begin
        setTrainingInput(s, $batchInput)
        computeAll_parallel(s)
    end evals = 1 samples = nSamples setup = (s = $tmp_struct)
    display(bmk_CG_all)
    @test bmk_CG_all.allocs <= 200 # computeAll_parallel allocates variables

    #rm("tests/tmp_struct_code.jl")
end

@testset "CodeGeneration: computeAsync_parallel" begin
    lossType = :sse # TODO `loss_grad_Enzyme` given error with :mse
    (graph, theta, inference, training, optimizerNodes...) = denseChain(TypeValue;
        nNodes,
        inferenceBatchSize=1,
        trainingBatchSize,
        loss=lossType, optimizer=:gradDescent)

    Revise.track(Main, "tests/tmp_struct_code.jl"; mode=:eval) # includet default is not good enough
    revise() # seems needed after track
    tmp_struct = Tmp_struct()

    # check sets
    setTheta(tmp_struct, W..., b...)
    set!(graph, values(theta), (W..., b...))
    @test theta.layer1_W.id == 1
    @test all(tmp_struct.Node1 .== W[1])
    @test all(tmp_struct.Node2 .== W[2])
    @test all(tmp_struct.Node3 .== W[3])
    @test theta.layer1_b.id == 4
    @test all(tmp_struct.Node4 .== b[1])
    @test all(tmp_struct.Node5 .== b[2])
    @test all(tmp_struct.Node6 .== b[3])

    setInferenceInput(tmp_struct, input)
    set!(graph, inference.input, input)
    @test inference.input.id == 7
    @test all(tmp_struct.Node7 .== input)

    setTrainingInput(tmp_struct, batchInput)
    set!(graph, training.input, batchInput)
    @test training.input.id == 8
    @test all(tmp_struct.Node8 .== batchInput)

    setTrainingReference(tmp_struct, batchReference)
    set!(graph, training.reference, batchReference)
    @test training.reference.id == 9
    @test all(tmp_struct.Node9 .== batchReference)

    setEta(tmp_struct, eta)
    set!(graph, optimizerNodes.eta, eta)
    @test optimizerNodes.eta.id == 26
    @test all(tmp_struct.Node26 .== eta)

    #@code_warntype setInferenceInput(tmp_struct, input)

    computeAsync_parallel(tmp_struct)
    println("Initial counts:")
    println(tmp_struct.cg_counts)
    @test maximum(tmp_struct.cg_counts) == 0
    fill!(tmp_struct.cg_counts, 0)

    @test training.loss.id == 25
    @test optimizerNodes.gradients[1].id == 38
    @test optimizerNodes.gradients[2].id == 34
    @test optimizerNodes.gradients[3].id == 30
    @test optimizerNodes.gradients[4].id == 40
    @test optimizerNodes.gradients[5].id == 41
    @test optimizerNodes.gradients[6].id == 42

    if false
        get_gradientAsync_parallel(tmp_struct)
        getLossAsync_parallel(tmp_struct)
        println("Counts after async gets:")
        println(tmp_struct.cg_counts)
        fill!(tmp_struct.cg_counts, 0)
    else
        @time for i = 1:nSamples
            setTrainingInput(tmp_struct, batchInput)
            notify(tmp_struct.Node25valid_needed)
            notify(tmp_struct.Node38valid_needed)
            notify(tmp_struct.Node34valid_needed)
            notify(tmp_struct.Node30valid_needed)
            notify(tmp_struct.Node40valid_needed)
            notify(tmp_struct.Node41valid_needed)
            notify(tmp_struct.Node42valid_needed)

            wait(tmp_struct.Node25valid)
            wait(tmp_struct.Node38valid)
            wait(tmp_struct.Node34valid)
            wait(tmp_struct.Node30valid)
            wait(tmp_struct.Node40valid)
            wait(tmp_struct.Node41valid)
            wait(tmp_struct.Node42valid)
        end
        println("Counts after loop of $(nSamples) notify+wait:")
        println(tmp_struct.cg_counts)
        @test maximum(tmp_struct.cg_counts) == nSamples
        fill!(tmp_struct.cg_counts, 0)
    end

    ## check correctness
    @test training.loss.id == 25
    @show loss = tmp_struct.Node25
    @show loss1 = get(graph, training.loss)
    @test norm(loss .- loss1) < 1e-8

    @test norm(get(graph, optimizerNodes.gradients[1]) .- tmp_struct.Node38) < 1e-8
    @test norm(get(graph, optimizerNodes.gradients[2]) .- tmp_struct.Node34) < 1e-8
    @test norm(get(graph, optimizerNodes.gradients[3]) .- tmp_struct.Node30) < 1e-8
    @test norm(get(graph, optimizerNodes.gradients[4]) .- tmp_struct.Node40) < 1e-8
    @test norm(get(graph, optimizerNodes.gradients[5]) .- tmp_struct.Node41) < 1e-8
    @test norm(get(graph, optimizerNodes.gradients[6]) .- tmp_struct.Node42) < 1e-8

    print(@bold @green "set(batchInput)+computeAsync_parallel(loss,grad): ")
    bmk_CG_input_loss_grad = @benchmark begin
        setTrainingInput(s, $batchInput)
        notify(s.Node25valid_needed)
        notify(s.Node38valid_needed)
        notify(s.Node34valid_needed)
        notify(s.Node30valid_needed)
        notify(s.Node40valid_needed)
        notify(s.Node41valid_needed)
        notify(s.Node42valid_needed)

        wait(s.Node25valid)
        wait(s.Node38valid)
        wait(s.Node34valid)
        wait(s.Node30valid)
        wait(s.Node40valid)
        wait(s.Node41valid)
        wait(s.Node42valid)
    end evals = 1 samples = nSamples setup = (s = $tmp_struct)
    display(bmk_CG_input_loss_grad)
    @test bmk_CG_input_loss_grad.allocs == 0
    println("Counts benchmark:")
    println(tmp_struct.cg_counts)
    fill!(tmp_struct.cg_counts, 0)

    print(@bold @green "set(batchInput)+getAsync_parallel(loss,grad): ")
    bmk_CG_input_loss_grad = @benchmark begin
        setTrainingInput(s, $batchInput)
        getLossAsync_parallel(s)
        get_gradientAsync_parallel(s)
    end evals = 1 samples = nSamples setup = (s = $tmp_struct)
    display(bmk_CG_input_loss_grad)
    @test bmk_CG_input_loss_grad.allocs == 0
    println("Counts benchmark:")
    println(tmp_struct.cg_counts)
    fill!(tmp_struct.cg_counts, 0)

    #rm("tests/tmp_struct_code.jl")
end
