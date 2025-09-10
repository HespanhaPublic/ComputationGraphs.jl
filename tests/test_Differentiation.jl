using Revise

using Flux

using ComputationGraphs

using LinearAlgebra
using Test

@testset "test: d(variable1,variable1)" begin
    g = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, -3.0]
    y_ = [4.0, 5.0, -6.0]
    x = variable(g, size(x_))
    @test length(g.nodes) == 1
    set!(g, x, x_)
    z = variable(g, rand(4, 3))
    @test length(g.nodes) == 2
    y = constant(g, y_)
    @test length(g.nodes) == 3

    yDxx = D(g, y, x, x)
    @test length(g.nodes) == 3 # it is just y

    compute!(g; force=true)
    display(g)
    yDxx_ = get(g, yDxx)

    @test length(g.nodes) == 3
    @test size(yDxx_) == size(x)
    @test norm(yDxx_ .- y_) < 1e-10
end

@testset "test: d(variable1,variable2)" begin
    g = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, -3.0]
    y_ = [4.0, 5.0, -6.0]
    x = variable(g, size(x_))
    @test length(g.nodes) == 1
    set!(g, x, x_)
    z = variable(g, rand(Float64, 4, 3))
    @test length(g.nodes) == 2
    y = constant(g, y_)
    @test length(g.nodes) == 3

    yDxz = D(g, y, x, z)
    @test length(g.nodes) == 4 # zero

    compute!(g; force=true)
    display(g)
    yDxz_ = get(g, yDxz)

    @test length(g.nodes) == 4
    @test size(yDxz_) == size(z)
    @test norm(yDxz_) < 1e-10

end

@testset "test: d(constant,variable)" begin
    g = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, -3.0]
    y_ = [4.0, 5.0, -6.0]
    x = variable(g, size(x_))
    @test length(g.nodes) == 1
    set!(g, x, x_)
    z = variable(g, rand(Float64, 3))
    @test length(g.nodes) == 2
    y = constant(g, y_)
    @test length(g.nodes) == 3

    yDxz = D(g, x, y, z)
    @test length(g.nodes) == 4 # zero

    compute!(g; force=true)
    display(g)
    yDxz_ = get(g, yDxz)

    @test length(g.nodes) == 4
    @test size(yDxz_) == size(z)
    @test norm(yDxz_) < 1e-10

end

@testset "test: d(zero,variable)" begin
    g = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, -3.0]
    x = variable(g, size(x_))
    @test length(g.nodes) == 1
    set!(g, x, x_)
    z = variable(g, rand(Float64, 4, 3))
    @test length(g.nodes) == 2
    y = zeros(g, size(x))
    @test length(g.nodes) == 3

    yDxz = D(g, x, y, z)
    @test length(g.nodes) == 4 # zero already exists, but different size

    compute!(g; force=true)
    display(g)
    yDxz_ = get(g, yDxz)

    @test length(g.nodes) == 4
    @test size(yDxz_) == size(z)
    @test norm(yDxz_) < 1e-10

end

@testset "test: d(norm2)" begin
    g = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, -3.0]
    x = variable(g, size(x_))
    set!(g, x, x_)
    nx2 = norm2(g, x)
    @test length(g.nodes) == 2

    d = D(g, nx2, x)

    compute!(g; force=true)
    display(g)
    d_ = get(g, d)

    @test length(g.nodes) == 4 # including constant
    @test size(d) == size(x)
    @test norm(d_ .- (2 * x_)) < 1e-10
end

@testset "test: d(times,A)" begin
    g = ComputationGraph{Float64}()

    A_ = [1.0 2.0 3.0; 4.0 5.0 6.0]
    x_ = [4.0, 5.0, 6.0]
    A = variable(g, size(A_))
    set!(g, A, A_)
    x = variable(g, size(x_))
    set!(g, x, x_)

    y = times(g, A, x)
    @test length(g.nodes) == 3

    v_ = [10.0, 100.0]
    v = constant(g, v_)
    @test length(g.nodes) == 4

    d = D(g, v, y, A)

    compute!(g; force=true)
    display(g)
    d_ = get(g, d)

    @test length(g.nodes) == 7
    @test size(d_) == size(A)
    @test norm(d_ .- (v_ * x_')) < 1e-10 # TODO verify
end

@testset "test: d(times,x)" begin
    g = ComputationGraph{Float64}()

    A_ = [1.0 2.0 3.0; 4.0 5.0 6.0]
    x_ = [4.0, 5.0, 6.0]
    A = variable(g, size(A_))
    set!(g, A, A_)
    x = variable(g, size(x_))
    set!(g, x, x_)

    y = times(g, A, x)
    @test length(g.nodes) == 3

    v_ = [10.0, 100.0]
    v = constant(g, v_)
    @test length(g.nodes) == 4

    d = D(g, v, y, x)

    compute!(g; force=true)
    display(g)
    d_ = get(g, d)

    @test length(g.nodes) == 7
    @test size(d_) == size(x)
    @test norm(d_ .- (A_' * v_)) < 1e-10 # TODO verify
end


@testset "test: d(affine,A)" begin
    g = ComputationGraph{Float64}()

    A_ = [1.0 2.0 3.0; 4.0 5.0 6.0]
    x_ = [4.0, 5.0, 6.0]
    b_ = [7.0, 8.0]
    A = variable(g, size(A_))
    set!(g, A, A_)
    x = variable(g, size(x_))
    set!(g, x, x_)
    b = variable(g, size(b_))
    set!(g, b, b_)

    y = affine(g, A, x, b)
    @test length(g.nodes) == 4

    v_ = [10.0, 100.0]
    v = constant(g, v_)
    @test length(g.nodes) == 5

    d = D(g, v, y, A)

    compute!(g; force=true)
    display(g)
    d_ = get(g, d)

    @test length(g.nodes) == 8
    @test size(d_) == size(A)
    @test norm(d_ .- (v_ * x_')) < 1e-10 # TODO verify
end

@testset "test: d(affine,x)" begin
    g = ComputationGraph{Float64}()

    A_ = [1.0 2.0 3.0; 4.0 5.0 6.0]
    x_ = [4.0, 5.0, 6.0]
    b_ = [7.0, 8.0]
    A = variable(g, size(A_))
    set!(g, A, A_)
    x = variable(g, size(x_))
    set!(g, x, x_)
    b = variable(g, size(b_))
    set!(g, b, b_)

    y = affine(g, A, x, b)
    @test length(g.nodes) == 4

    v_ = [10.0, 100.0]
    v = constant(g, v_)
    @test length(g.nodes) == 5

    d = D(g, v, y, x)

    compute!(g; force=true)
    display(g)
    d_ = get(g, d)

    @test length(g.nodes) == 8
    @test size(d_) == size(x)
    @test norm(d_ .- (A_' * v_)) < 1e-10 # TODO verify
end

@testset "test: d(affine,b)" begin
    g = ComputationGraph{Float64}()

    A_ = [1.0 2.0 3.0; 4.0 5.0 6.0]
    x_ = [4.0, 5.0, 6.0]
    b_ = [7.0, 8.0]
    A = variable(g, size(A_))
    set!(g, A, A_)
    x = variable(g, size(x_))
    set!(g, x, x_)
    b = variable(g, size(b_))
    set!(g, b, b_)

    y = affine(g, A, x, b)
    @test length(g.nodes) == 4

    v_ = [10.0, 100.0]
    v = constant(g, v_)
    @test length(g.nodes) == 5

    d = D(g, v, y, b)

    compute!(g; force=true)
    display(g)
    d_ = get(g, d)

    @test length(g.nodes) == 8
    @test size(d_) == size(b)
    @test norm(d_ .- v_) < 1e-10 # TODO verify
end

@testset "test: d(x-y,x)" begin
    g = ComputationGraph{Float64}()

    x_ = [4.0, 5.0, 6.0]
    x = variable(g, size(x_))
    set!(g, x, x_)
    y_ = [4.0, 5.0, 6.0]
    y = variable(g, size(y_))
    set!(g, y, y_)

    z = -(g, x, y)
    @test length(g.nodes) == 3

    v_ = [10.0, 100.0, 1000.0]
    v = constant(g, v_)
    @test length(g.nodes) == 4

    d = D(g, v, z, x)

    compute!(g; force=true)
    display(g)
    d_ = get(g, d)

    @test length(g.nodes) == 5
    @test size(d_) == size(x)
    @test norm(d_ .- v_) < 1e-10 # TODO verify
end

@testset "test: d(x-y,y)" begin
    g = ComputationGraph{Float64}()

    x_ = [4.0, 5.0, 6.0]
    x = variable(g, size(x_))
    set!(g, x, x_)
    y_ = [4.0, 5.0, 6.0]
    y = variable(g, size(y_))
    set!(g, y, y_)

    z = -(g, x, y)
    @test length(g.nodes) == 3

    v_ = [10.0, 100.0, 1000.0]
    v = constant(g, v_)
    @test length(g.nodes) == 4

    d = D(g, v, z, y)

    compute!(g; force=true)
    display(g)
    d_ = get(g, d)

    @test length(g.nodes) == 6
    @test size(d_) == size(y)
    @test norm(d_ .+ v_) < 1e-10 # TODO verify
end

@testset "test: d(relu,x)" begin
    g = ComputationGraph{Float64}()

    x_ = [4.0, 5.0, -6.0, 0.0]
    x = variable(g, size(x_))
    set!(g, x, x_)

    z = ComputationGraphs.relu(g, x)
    @test length(g.nodes) == 2

    v_ = [10.0, 100.0, 1000.0, 10000.0]
    v = constant(g, v_)
    @test length(g.nodes) == 3

    d = D(g, v, z, x)

    compute!(g; force=true)
    display(g)
    d_ = get(g, d)

    @test length(g.nodes) == 5
    @test size(d_) == size(x)
    @test norm(d_ .- (v_ .* heaviside(x_))) < 1e-10 # TODO verify
end