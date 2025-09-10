using Revise

using Flux
using LinearAlgebra

using ComputationGraphs

using Test

@testset "test: relu(constant)" begin
    g = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, -3.0, 0.0]
    x = constant(g, x_)

    y = ComputationGraphs.relu(g, x)
    @time compute!(g; force=true)
    display(g)
    @show y_ = get(g, y)

    @test length(g.nodes) == 2
    @test norm(y_ .- Flux.relu(x_)) < 1e-10
end

@testset "test: relu(variable)" begin
    g = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, -3.0, 0.0]
    x = variable(g, size(x_))
    set!(g, x, x_)

    y = ComputationGraphs.relu(g, x)
    @time compute!(g; force=true)
    display(g)
    @show y_ = get(g, y)

    @test length(g.nodes) == 2
    @test norm(y_ .- Flux.relu(x_)) < 1e-10
end

@testset "test: heaviside(constant)" begin
    g = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, -3.0, 0.0]
    x = constant(g, x_)

    y = ComputationGraphs.heaviside(g, x)
    @time compute!(g; force=true)
    display(g)
    @show y_ = get(g, y)

    @test length(g.nodes) == 2
    @test norm(y_ .- convert(Vector{Float64}, x_ .> 0)) < 1e-10
end


@testset "test: sign(constant)" begin
    g = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, -3.0, 0.0, 0.1, -0.3]
    x = constant(g, x_)

    y = ComputationGraphs.sign(g, x)
    @time compute!(g; force=true)
    display(g)
    @show y_ = get(g, y)

    @test length(g.nodes) == 2
    @test norm(y_ .- convert(Vector{Float64}, sign.(x_))) < 1e-10
end

@testset "test: sat(constant)" begin
    g = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, -3.0, 0.0, 0.1, -0.3]
    x = constant(g, x_)

    y = ComputationGraphs.sat(g, x)
    @time compute!(g; force=true)
    display(g)
    @show y_ = get(g, y)

    @test length(g.nodes) == 2
    @test norm(y_ .- convert(Vector{Float64}, min.(max.(x_, -1), 1))) < 1e-10
end
