using Revise

using LinearAlgebra
using BenchmarkTools

using ComputationGraphs

using Test

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 3
@testset "test_LinearAlgebra: norm2(constant)" begin
    graph = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, 3.0]
    x = constant(graph, x_)

    y = norm2(graph, x)

    @test graph.nodes[1].parentIds == ()
    @test graph.nodes[2].parentIds == (x.id,)

    @test graph.parents == [[], []] # constants do not appear in parents lists

    # @code_warntype compute!(x)

    compute!(graph; force=true)
    display(graph)
    @show y_ = get(graph, y)

    @test length(graph.nodes) == 2
    @test norm(y_ .- norm(x_)^2) < 1e-10

    bmk = @benchmark compute!($graph, $y)
    display(bmk)
    @test bmk.allocs == 0

    bmk = @benchmark compute!($graph; force=true)
    display(bmk)
    @test bmk.allocs == 0
end

@testset "test_LinearAlgebra: norm2(variable)" begin
    graph = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, 3.0]
    x = variable(graph, size(x_))
    set!(graph, x, x_)

    #@code_warntype compute!(x)

    y = norm2(graph, x)
    compute!(graph; force=true)
    display(graph)
    @show y_ = get(graph, y)

    @test length(graph.nodes) == 2
    @test norm(y_ .- norm(x_)^2) < 1e-10
end

@testset "test_LinearAlgebra: adjoint" begin
    graph = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, 3.0]
    x = constant(graph, x_)
    y = adjoint(graph, x)
    compute!(graph; force=true)
    y_ = get(graph, y)

    @test length(graph.nodes) == 2
    @test norm(y_ .- x_') < 1e-10

    x_ = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    x = constant(graph, x_)
    y = adjoint(graph, x)
    compute!(graph; force=true)
    y_ = get(graph, y)

    @test length(graph.nodes) == 4
    @test norm(y_ .- x_') < 1e-10

    z = adjoint(graph, y)
    @test length(graph.nodes) == 4
    compute!(graph; force=true)
    z_ = get(graph, z)
    @test norm(z_ .- x_) < 1e-10
end

@testset "test_LinearAlgebra: column/sumColumns" begin
    graph = ComputationGraph{Float64}()

    A_ = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    A = constant(graph, A_)

    y = sumColumns(graph, A)

    compute!(graph; force=true)
    y_ = get(graph, y)

    @test length(graph.nodes) == 2
    @test norm(y_ .- sum(A_, dims=2)) < 1e-10

    a_ = [1.0, 2.0, 3.0, 4.0]
    a = constant(graph, a_)
    rows_ = [1, 1, 2, 2]
    rows = constant(graph, rows_)

    nRows = 4
    z = sumExpandColumns(graph, a, rows, nRows)

    compute!(graph; force=true)
    z_ = get(graph, z)

    @show aExp = expandColumns(a_, rows_, nRows)

    @test length(graph.nodes) == 5
    @test norm(z_ .- sum(aExp, dims=2)) < 1e-10
end

@testset "test_LinearAlgebra: +(constant,zero)" begin
    graph = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, 3.0]
    x = constant(graph, x_)
    y = zeros(graph, size(x))
    @test length(graph.nodes) == 2


    z1 = +(graph, x, y)
    z2 = +(graph, y, x)
    @test length(graph.nodes) == 2

    #@code_warntype compute!(z1)
    #@code_warntype compute!(z2)
    compute!(graph; force=true)
    display(graph)

    @show z1_ = get(graph, z1)
    @show z2_ = get(graph, z2)

    @test length(graph.nodes) == 2 # just zero and x, since none of the summations matters
    @test norm(z1_ .- x_) < 1e-10
    @test norm(z2_ .- x_) < 1e-10
end

@testset "test_LinearAlgebra: +(constant,constant)" begin
    graph = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, 3.0]
    y_ = [4.0, 5.0, 6.0]
    x = constant(graph, x_)
    y = constant(graph, y_)
    @test length(graph.nodes) == 2

    z = +(graph, x, y)
    @test length(graph.nodes) == 3
    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 3
    @test norm(z_ .- (x_ + y_)) < 1e-10

    # using broadcast
    z = broadcast(+, graph, x, y) # same
    @test length(graph) == 3

    @macroexpand @add graph x .+ y #same 
    z = @add graph x .+ y
    @test length(graph) == 3
end

@testset "test_LinearAlgebra: -(constant,constant)" begin
    graph = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, 3.0]
    y_ = [4.0, 5.0, 6.0]
    x = constant(graph, x_)
    y = constant(graph, y_)

    z = -(graph, x, y)
    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 3
    @test norm(z_ .- (x_ - y_)) < 1e-10

    # using broadcast
    z = broadcast(-, graph, x, y) # same
    @test length(graph) == 3

    @macroexpand @add graph x .+ y #same 
    z = @add graph x .- y
    @test length(graph) == 3
end

@testset "test_LinearAlgebra: scalarPlus" begin
    graph = ComputationGraph{Float64}()

    x_ = 1.0
    y_ = [4.0, 5.0, 6.0]
    x = constant(graph, x_)
    y = constant(graph, y_)

    z = scalarPlus(graph, x, y)
    @test length(graph) == 3
    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 3
    @test norm(z_ .- (x_ .+ y_)) < 1e-10

    # using broadcast
    z = broadcast(+, graph, x, y) # same
    @test length(graph) == 3

    @macroexpand @add graph x .+ y #same 
    z = @add graph x .+ y
    @test length(graph) == 3
end

@testset "test_LinearAlgebra: columnPlus" begin
    graph = ComputationGraph{Float64}()

    x_ = [1.0, 2.0]
    y_ = [4.0 5.0 6.0; 7.0 8.0 9.0]
    x = constant(graph, x_)
    y = constant(graph, y_)

    z = columnPlus(graph, x, y)
    @test length(graph) == 3
    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 3
    @test norm(z_ .- (x_ .+ y_)) < 1e-10

    # using broadcast
    z = broadcast(+, graph, x, y) # same
    @test length(graph) == 3

    @macroexpand @add graph x .+ y #same 
    z = @add graph x .+ y
    @test length(graph) == 3

    # using broadcast
    z = broadcast(+, graph, y, x) # same
    @test length(graph) == 3

    @macroexpand @add graph y .+ x #same 
    z = @add graph x .+ y
    @test length(graph) == 3
end

@testset "test_LinearAlgebra: rowPlus" begin
    graph = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, 3.0]
    y_ = [4.0 5.0 6.0; 7.0 8.0 9.0]
    x = constant(graph, x_)
    y = constant(graph, y_)

    z = rowPlus(graph, x, y)
    @test length(graph) == 3
    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 3
    @test norm(z_ .- (x_' .+ y_)) < 1e-10

    # using broadcast
    z = broadcast(+, graph, x, y) # same
    @test length(graph) == 3

    @macroexpand @add graph x .+ y #same 
    z = @add graph x .+ y
    @test length(graph) == 3

    # using broadcast
    z = broadcast(+, graph, y, x) # same
    @test length(graph) == 3

    @macroexpand @add graph y .+ x #same 
    z = @add graph x .+ y
    @test length(graph) == 3
end

@testset "test_LinearAlgebra: scalarTimes(constant,constant)" begin
    graph = ComputationGraph{Float64}()

    x_ = 2.0
    y_ = [4.0, 5.0, 6.0]
    x = constant(graph, x_)
    y = constant(graph, y_)

    z = scalarTimes(graph, x, y)
    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 3
    @test norm(z_ .- (x_ .* y_)) < 1e-10

    # broadcast
    z = broadcast(*, graph, x, y) # same
    @test length(graph) == 3

    @macroexpand @add graph x .* y #same 
    z = @add graph x .* y
    @test length(graph) == 3

    z = broadcast(*, graph, y, x) # same
    @test length(graph) == 3

    @macroexpand @add graph y .* x #same 
    z = @add graph x .* y
    @test length(graph) == 3
end

@testset "test_LinearAlgebra: pointTimes(constant,constant) & pointDivision" begin
    graph = ComputationGraph{Float64}()

    x_ = [1.0, 2.0, 3.0]
    y_ = [4.0, 5.0, 6.0]
    x = constant(graph, x_)
    y = constant(graph, y_)

    z = pointTimes(graph, x, y)
    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 3
    @test norm(z_ .- (x_ .* y_)) < 1e-10

    z = pointDivide(graph, x, y)
    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 4
    @test norm(z_ .- (x_ ./ y_)) < 1e-10

    # broadcast
    z = broadcast(*, graph, x, y) # same
    @test length(graph) == 4

    @macroexpand @add graph x .* y #same 
    z = @add graph x .* y
    @test length(graph) == 4
end

@testset "test_LinearAlgebra: times(constant,constant)" begin
    graph = ComputationGraph{Float64}()

    x_ = [1.0 2.0 3.0; 4.0 5.0 6.0]
    y_ = [4.0, 5.0, 6.0]
    x = constant(graph, x_)
    y = constant(graph, y_)

    z = times(graph, x, y)
    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 3
    @test norm(z_ .- (x_ * y_)) < 1e-10

    x_ = [1.0 2.0 3.0; 4.0 5.0 6.0]
    y_ = [4.0 8.0; 5.0 1.0; 6.0 9.0]
    x = constant(graph, x_)
    y = constant(graph, y_)

    z = times(graph, x, y)
    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 5
    @test norm(z_ .- (x_ * y_)) < 1e-10

    o = ones(graph, size(x_, 2))
    z = times(graph, x, o)
    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 7
    @test isa(z, ComputationGraphs.NodeSumColumns)
    @test norm(z_ .- sum(x_, dims=2)) < 1e-10
end


@testset "test_LinearAlgebra: adjointTimes(constant,constant)" begin
    graph = ComputationGraph{Float64}()

    x_ = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    y_ = [4.0, 5.0, 6.0]
    x = constant(graph, x_)
    y = constant(graph, y_)

    z = adjointTimes(graph, x, y)
    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 3
    @test norm(z_ .- (x_' * y_)) < 1e-10

    x_ = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    y_ = [4.0 8.0; 5.0 1.0; 6.0 9.0]
    x = constant(graph, x_)
    y = constant(graph, y_)

    z = adjointTimes(graph, x, y)
    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 5
    @test norm(z_ .- (x_' * y_)) < 1e-10

    a_ = [1.0, 2.0, 3.0]
    a = constant(graph, a_)
    rows_ = [1, 2, 1]
    rows = constant(graph, rows_)
    aExp = expandColumns(a_, rows_, size(x, 1))
    #display(x_')
    #display(aExp)
    #display(x_' * aExp)
    z = adjointTimesExpandColumns(graph, x, a, rows)
    compute!(graph; force=true)
    display(graph)

    @show z_ = get(graph, z)
    @test length(graph.nodes) == 8
    @test norm(z_ .- (x_' * aExp)) < 1e-10
end

@testset "test_LinearAlgebra: timesAdjoint(constant,constant)" begin
    graph = ComputationGraph{Float64}()

    # y vector
    x_ = [1.0, 2.0, 3.0]
    y_ = [4.0, 5.0, 6.0]
    x = constant(graph, x_)
    y = constant(graph, y_)

    z = timesAdjoint(graph, x, y)
    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 3
    @test norm(z_ .- (x_ * y_')) < 1e-10

    # x,y matrix
    x_ = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    y_ = [4.0 5.0; 6.0 7.0; 8.0 9.0]
    x = constant(graph, x_)
    y = constant(graph, y_)

    z = timesAdjoint(graph, x, y)
    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 6
    @test norm(z_ .- (x_ * y_')) < 1e-10

    # expandColumns
    a_ = [4.0, 5.0]
    a = constant(graph, a_)
    nRows = 3
    rows_ = [2, 1]
    rows = constant(graph, rows_)
    aExp = expandColumns(a_, rows_, nRows)

    z = expandColumnsTimesAdjoint(graph, a, y, rows, nRows)

    compute!(graph; force=true)
    display(graph)

    @show z_ = get(graph, z)
    @test length(graph.nodes) == 9
    @test norm(z_ .- (aExp * y_')) < 1e-10
end

@testset "test_LinearAlgebra: timesAdjoint special" begin
    graph = ComputationGraph{Float64}()

    x_ = [1.0, 0.0, 0.0]
    y_ = [4.0, 5.0, 6.0]
    x = unitvector(graph, (3,), 1)
    y = constant(graph, y_)

    z = timesAdjoint(graph, x, y)
    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 3
    @test norm(z_ .- (x_ * y_')) < 1e-10

    o_ = ones(Float64, 5)
    o = ones(graph, 5)
    w = timesAdjoint(graph, y, o)
    @show typeof(w)
    compute!(graph; force=true)
    display(graph)
    @show w_ = get(graph, w)

    @test length(graph.nodes) == 5
    @test norm(w_ .- (y_ * o_')) < 1e-10
end

@testset "test_LinearAlgebra: division and power vector-scalar" begin
    graph = ComputationGraph{Float64}()

    s_ = 3.0
    s = constant(graph, s_)

    x_ = [1 2 3; 4 5 6.0]
    x = constant(graph, x_)

    y = divideScalar(graph, x, s)

    compute!(graph; force=true)
    display(graph)
    @show y_ = get(graph, y)

    @test length(graph.nodes) == 3
    @test norm(y_ .- (x_ ./ s_)) < 1e-10

    y = scalarDivide(graph, s, x)

    compute!(graph; force=true)
    display(graph)
    @show y_ = get(graph, y)

    @test length(graph.nodes) == 4
    @test norm(y_ .- (s_ ./ x_)) < 1e-10

    y = ^(graph, x, s)

    compute!(graph; force=true)
    display(graph)
    @show y_ = get(graph, y)

    @test length(graph.nodes) == 5
    @test norm(y_ .- (x_ .^ s_)) < 1e-10
end

@testset "test_LinearAlgebra: affine(constant,constant,constant)" begin
    graph = ComputationGraph{Float64}()

    A_ = [1.0 2.0 3.0; 4.0 5.0 6.0]
    x_ = [4.0, 5.0, 6.0]
    b_ = [7.0, 8.0]
    A = constant(graph, A_)
    x = constant(graph, x_)
    b = constant(graph, b_)

    z = affine(graph, A, x, b)

    @test graph.nodes[1].parentIds == ()
    @test graph.nodes[2].parentIds == ()
    @test graph.nodes[3].parentIds == ()
    @test graph.nodes[4].parentIds == (A.id, x.id, b.id)

    @test graph.parents == [[], [], [], []] # constants do not appear in parents lists

    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 4
    @test norm(z_ .- (A_ * x_ + b_)) < 1e-10

    bmk = @benchmark compute!($graph, $z)
    display(bmk)
    @test bmk.allocs == 0

    valid = graph.validValue
    id = z.id
    #f = graph.computeFunctions[id]
    bmk = @benchmark begin
        $valid[$id] = false
        compute!($graph, $z)
        #$f($graph)
    end
    display(bmk)
    @test bmk.allocs == 0
end

@testset "test_LinearAlgebra: affineRows(constant,constant,constant,rows)" begin
    graph = ComputationGraph{Float64}()

    A_ = [1.0 2.0 3.0; 4.0 5.0 6.0]
    x_ = [4.0, 5.0, 6.0]
    b_ = [7.0, 8.0]
    A = constant(graph, A_)
    x = constant(graph, x_)
    b = constant(graph, b_)

    rows_ = [1, 1]
    rows = variable(graph, Int64, rows_)
    set!(graph, rows, rows_)
    z = affineRows(graph, A, x, b, rows)

    compute!(graph; force=true)
    display(graph)
    @show z_ = get(graph, z)

    @test length(graph.nodes) == 5
    @test norm(z_ .- (A_*x_+b_)[rows_, :]) < 1e-10

    bmk = @benchmark compute!($graph, $z)
    display(bmk)
    @test bmk.allocs == 0

    valid = graph.validValue
    id = z.id
    #f = graph.computeFunctions[id]
    bmk = @benchmark begin
        $valid[$id] = false
        compute!($graph, $z)
        #$f($graph)
    end
    display(bmk)
    @test bmk.allocs == 0
end

@testset "test_LinearAlgebra: maxRow, findMax" begin
    graph = ComputationGraph{Float64}()

    @show A_ = [1.0 5.0 3.0; 4.0 2.0 6.0]
    A = constant(graph, A_)

    z = maxRow(graph, A)

    @test length(graph.nodes) == 2
    compute!(graph; force=true)
    display(graph)

    @show z_ = get(graph, z)
    @test norm(z_ .- dropdims(maximum(A_, dims=1), dims=1)) < 1e-10

    bmk = @benchmark compute!($graph, $z)
    display(bmk)
    @test bmk.allocs == 0

    z = findMaxRow(graph, A)

    @test length(graph.nodes) == 3
    compute!(graph; force=true)
    display(graph)

    @show z_ = get(graph, z)
    @show maxi = vec([c[1] for c in findmax(A_, dims=1)[2]])
    @test all(z_ .== maxi)

    bmk = @benchmark compute!($graph, $z)
    display(bmk)
    @test bmk.allocs == 0

end

@testset "test_LinearAlgebra: selectRows, expandColumns" begin
    graph = ComputationGraph{Float64}()

    # matrix
    @show A_ = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    A = constant(graph, A_)

    rows_ = [1, 3]
    rows = variable(graph, rows_)
    set!(graph, rows, rows_)
    z = selectRows(graph, A, rows)

    @test length(graph.nodes) == 3
    compute!(graph; force=true)
    display(graph)

    @show z_ = get(graph, z)
    @test norm(z_ .- [A_[r, i] for (i, r) in enumerate(rows_)]) < 1e-10

    # vector
    @show a_ = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    a = constant(graph, a_)

    rows_ = [1, 1, 4, 3, 2, 1]
    nRows = 5
    rows = variable(graph, rows_)
    set!(graph, rows, rows_)
    z = expandColumns(graph, a, rows, nRows)
    zTest = expandColumns(a_, rows_, nRows)

    @test length(graph.nodes) == 6
    compute!(graph; force=true)
    display(graph)

    @show z_ = get(graph, z)
    @test size(z_) == (5, length(rows_))
    @test all(z_[r, i] == a_[i] for (i, r) in enumerate(rows_))
end

@testset "test_LinearAlgebra: @add" begin
    graph = ComputationGraph{Float64}()

    A_ = [1.0 2.0 3.0; 4.0 5.0 6.0]
    x_ = [4.0, 5.0, 6.0]
    b_ = [7.0, 8.0]

    A = constant(graph, A_)
    x = constant(graph, x_)
    b = constant(graph, b_)

    println(@macroexpand @add graph A * x + b)

    y = @add graph times(A, x) + b

    display(graph)
    @test length(graph.nodes) == 5

    z = @add graph ComputationGraphs.relu(times(A, x) + b) # will reuse previous nodes

    display(graph)
    @test length(graph.nodes) == 6
end
