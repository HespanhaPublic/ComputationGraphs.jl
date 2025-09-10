using Revise

using ComputationGraphs

using LinearAlgebra
using Test

@testset "test: unary operators" begin
    g = ComputationGraph{Float64}()

    a = variable(g, 3, 4)
    z1 = zeros(g, 3, 4)

    # identity
    aa = identity(g, a)
    ae = exp(g, a)
    @test a === aa

    aa = @add g identity(a)
    ae = @add g exp(a)
    @test a === aa

    # transpose of zero
    z2 = adjoint(g, z1)
    z3 = zeros(g, 4, 3)
    @test z2 === z3

    z2 = @add g z1'
    @test z2 === z3

    # transpose of transpose
    at = adjoint(g, a)
    att = adjoint(g, at)
    @test a === att

    at = @add g a'
    att = @add g at'
    @test a === att
end

@testset "test: sums & subtractions" begin
    g = ComputationGraph{Float64}()

    a = variable(g, 3, 4)
    z1 = zeros(g, 3, 4)

    # sum with zero
    az = +(g, a, z1)
    @test a === az
    az = @add g a + z1
    @test a === az

    az = +(g, z1, a)
    @test a === az
    az = @add g z1 + a
    @test a === az

    # unary sum
    as = +(g, a)
    @test as === a
    as = @add g +a
    @test as === a

    # subtraction of zero
    as = -(g, a, z1)
    @test as === a
    as = @add g a - z1
    @test as === a

    # zero subtraction
    as = -(g, z1, a)
    am = -(g, a)
    @test as === am

    as = @add g z1 - a
    am = @add g -a
    @test as === am
end

@testset "test: multiplication by 0" begin
    # TODO a lot of tests missing
end

@testset "test: multiplication by 1" begin
    g = ComputationGraph{Float64}()

    # matrix * ones
    A = variable(g, 3, 4)
    o = ones(g, 4)

    Ao = times(g, A, o)
    As = sumColumns(g, A)
    @test Ao === As

    Ao = @add g times(A, o)
    @test Ao === As

    # vector * ones'
    A = variable(g, 3)
    o = ones(g, 1)
    Ao = timesAdjoint(g, A, o)

    # TODO a lot of tests missing
end