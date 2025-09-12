using Revise

using LinearAlgebra
using ComputationGraphs

using Test
using BenchmarkTools
using Unrolled

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 3
@testset "test_Variable: variable" begin
    g = ComputationGraph(Float64)

    x_ = [1.0, 2.0, -3.0, 0.0]
    x = variable(g, size(x_))

    try
        compute!(x; force=false)
        @test false   # should raise exception for trying to get variable that has not been set
    catch
        @test true
    end

    try
        compute!(g; force=true)
        @test false   # should raise exception for trying to get variable that has not been set
    catch
        @test true
    end

    try
        y_ = get(g, x)
        @test false   # should raise exception for trying to get variable that has not been set
    catch
        @test true
    end

    set!(g, x, x_)
    @test g.nodes[1].parentIds == ()
    @test g.parents == [[]]
    @test g.children == [[]]

    display(g)
    @show y_ = get(g, x)

    @test length(g.nodes) == 1
    @test norm(y_ .- x_) < 1e-10

    set!(g, x, 2 * x_)

    display(g)
    @show y_ = get(g, x)

    # add another with same size (should not reuse)
    x = variable(g, size(x_))
    @test g.nodes[2].parentIds == ()
    @test g.parents == [[], []]
    @test g.children == [[], []]

    @test length(g.nodes) == 2
    @test norm(y_ .- 2 * x_) < 1e-10

    x = variable(g, 3 * x_)
    display(g)
    @show y_ = get(g, x)

    @test length(g.nodes) == 3
    @test norm(y_ .- 3 * x_) < 1e-10

    # copyto
    z = variable(g, size(x_))

    copyto!(g, z, x)
    display(g)
    @show z_ = get(g, z)

    @test length(g.nodes) == 4
    @test norm(z_ .- 3 * x_) < 1e-10

    # copyto (graph)
    z = variable(g, size(x_))

    copyto!(g, z, x)
    display(g)
    @show z_ = get(g, z)

    @test length(g.nodes) == 5
    @test norm(z_ .- 3 * x_) < 1e-10
end

@testset "test_Variable: constant" begin
    g = ComputationGraph(Float64)

    x_ = [1.0, 2.0, -3.0, 0.0]
    x = constant(g, x_)
    @test g.nodes[1].parentIds == ()

    @test g.parents == [[]]
    @test g.children == [[]]

    display(g)
    @show y_ = get(g, x)

    @test length(g.nodes) == 1
    @test norm(y_ .- x_) < 1e-10

    # add same, should reuse
    x = constant(g, x_)
    @show y_ = get(g, x)
    @test g.nodes[1].parentIds == ()

    @test g.parents == [[]]
    @test g.children == [[]]

    @test length(g.nodes) == 1
    @test norm(y_ .- x_) < 1e-10

    # add different
    x = constant(g, 2 * x_)
    @test g.nodes[2].parentIds == ()

    @test g.parents == [[], []]
    @test g.children == [[], []]

    display(g)
    @show y_ = get(g, x)
    @test length(g.nodes) == 2
    @test norm(y_ .- 2 * x_) < 1e-10

    # mismatch in types
    z_ = Float32[2.0, 4.0]
    @test_logs (:warn, "constant: value type=Float32 does not match graph's default type Float64, you can explicitly include the type in constant() to avoid this warning") z1 = constant(g, z_)

    @test_logs z2 = constant(g, Float32, z_)

    try
        z2 = constant(g, Float64, z_)
        @test false
    catch
        @test true
    end
end

@testset "test_Variable: zeros" begin

    g = ComputationGraph(Float64)

    x = zeros(g, (1,))
    @test g.nodes[1].parentIds == ()
    @test g.parents == [[]]
    @test g.children == [[]]

    @test length(g.nodes) == 1
    @test nodeValue(x) == zeros(Float64, 1)

    x = zeros(g, (1, 3))

    @test length(g.nodes) == 2
    @test nodeValue(x) == zeros(Float64, 1, 3)

    x = zeros(g, ())

    @test length(g.nodes) == 3
    @test nodeValue(x) == fill(zero(Float64))

end

@testset "test_Variable: ones" begin

    g = ComputationGraph(Float64)

    x = ones(g, (1,))
    @test g.nodes[1].parentIds == ()
    @test g.parents == [[]]
    @test g.children == [[]]

    @test length(g.nodes) == 1
    @test nodeValue(x) == ones(Float64, 1)

    x = ones(g, (1, 3))

    @test length(g.nodes) == 2
    @test nodeValue(x) == ones(Float64, 1, 3)

    x = ones(g, ())

    @test length(g.nodes) == 3
    @test nodeValue(x) == fill(one(Float64))

end

@testset "test_Variable: ones" begin

    g = ComputationGraph(Float64)

    x = unitvector(g, (4,), 2)
    @test g.nodes[1].parentIds == ()
    @test g.parents == [[]]
    @test g.children == [[]]

    @test nodeValue(x) == Float64[0.0, 1.0, 0.0, 0.0]

    @test unitvector(Float64, (4,), 3) == Float64[0.0, 0.0, 1.0, 0.0]
end

@testset "test_Variable: allocations for single node set() & copyto!()" begin
    g = ComputationGraph(Float64)

    x_ = ones(Float64, 1000)
    x = variable(g, size(x_))

    # set_node!
    @time ComputationGraphs.set_node!(x, x_)

    bmk = @benchmark ComputationGraphs.set_node!($x, $x_)
    display(bmk)
    @test bmk.allocs == 0

    s = variable(g, ())
    bmk = @benchmark for t_ in 1:10
        set!($g, $s, Float64(t_))
    end
    display(bmk)
    @test bmk.allocs == 0


    # set(graph,node)
    @time set!(g, x, x_)

    bmk = @benchmark set!($g, $x, $x_)
    display(bmk)
    @test bmk.allocs == 0

    # copyto_node!(node1,node2)
    y = variable(g, size(x_))
    @time ComputationGraphs.copyto_node!(x, y)

    bmk = @benchmark ComputationGraphs.copyto_node!($x, $y)
    display(bmk)
    @test bmk.allocs == 0

    # set(graph,variable)
    @time set!(g, x, x_)

    bmk = @benchmark set!($g, $x, $x_)
    display(bmk)
    @test bmk.allocs == 0

    # get(graph,variable)
    @time y_ = get(g, x)

    display(g)
    bmk = @benchmark get($g, $x)
    display(bmk)
    @test bmk.allocs == 0

    y = variable(g, size(x_))
    display(g)
    bmk = @benchmark copyto!($g, $y, $x)
    display(bmk)
    @test bmk.allocs == 0
end

@testset "test_Variable: allocations for a tuples of nodes set(), compute(), copyto!()" begin
    g = ComputationGraph(Float64)

    x1_ = 1.0:1.0:1000.0
    x1 = variable(g, size(x1_))
    x2_ = -x1_
    x2 = variable(g, size(x2_))
    x3_ = 2 * x1_
    x3 = variable(g, size(x3_))

    y1 = @add g x1 + x2
    y2 = @add g x1 - x2
    y3 = @add g x2 - x3

    x = (x1, x2, x3)
    x_ = (x1_, x2_, x3_)
    y = (y1, y2, y3)

    display(g)

    # set(graph,tuple)
    @time set!(g, x, x_)

    @test all(nodeValue(x1) == x1_)
    @test all(nodeValue(x2) == x2_)
    @test all(nodeValue(x3) == x3_)

    bmk = @benchmark set!($g, $x, $x_)
    display(bmk)
    @test bmk.allocs == 0

    # compute(graph,tuple)
    @time compute!(g, y)

    @test all(nodeValue(y1) == x1_ + x2_)
    @test all(nodeValue(y2) == x1_ - x2_)
    @test all(nodeValue(y3) == x2_ - x3_)

    bmk = @benchmark begin
        set!($g, $x, $x_)
        compute!($g, $y)
    end
    display(bmk)
    @test bmk.allocs == 0

    # copyto!(graph,tuple,tuple)

    @time copyto!(g, x, y)

    @test all(nodeValue(x1) == x1_ + x2_)
    @test all(nodeValue(x2) == x1_ - x2_)
    @test all(nodeValue(x3) == x2_ - x3_)

    #@code_unrolled Base.copyto!(g, x, y) # FIXME not working
    bmk = @benchmark copyto!($g, $x, $y)
    display(bmk)
    @test bmk.allocs == 0

    display(g)

    # get(graph,tuple)
    @time y_ = get(g, x)

    (y1_, y2_, y3_) = y_
    @test all(y1_ == x1_ + x2_)
    @test all(y2_ == x1_ - x2_)
    @test all(y3_ == x2_ - x3_)

    bmk = @benchmark get($g, $x)
    display(bmk)
    @test bmk.allocs <= 3 # FIXME
end

@testset "test_Variable: allocations for a named-tuple of nodes set(), compute(), copyto!()" begin
    g = ComputationGraph(Float64)

    x1_ = 1.0:1.0:1000.0
    x1 = variable(g, size(x1_))
    x2_ = -x1_
    x2 = variable(g, size(x2_))
    x3_ = 2 * x1_
    x3 = variable(g, size(x3_))

    y1 = @add g x1 + x2
    y2 = @add g x1 - x2
    y3 = @add g x2 - x3

    x = (a=x1, b=x2, c=x3)
    x_ = (a=x1_, b=x2_, c=x3_)    # set! requires the same order
    y = (a=y1, b=y2, c=y3)        # copyto! requires the same order

    display(g)

    # set(graph,tuple)
    @time set!(g, x, x_)

    @test all(nodeValue(x1) == x1_)
    @test all(nodeValue(x2) == x2_)
    @test all(nodeValue(x3) == x3_)

    bmk = @benchmark set!($g, $x, $x_)
    display(bmk)
    @test bmk.allocs == 0

    # compute(graph,tuple)
    @time compute!(g, y)

    @test all(nodeValue(y1) == x1_ + x2_)
    @test all(nodeValue(y2) == x1_ - x2_)
    @test all(nodeValue(y3) == x2_ - x3_)

    bmk = @benchmark begin
        set!($g, $x, $x_)
        compute!($g, $y)
    end
    display(bmk)
    @test bmk.allocs == 0

    # copyto!(graph,tuple,tuple)

    @time copyto!(g, x, y)

    @test all(nodeValue(x1) == x1_ + x2_)
    @test all(nodeValue(x2) == x1_ - x2_)
    @test all(nodeValue(x3) == x2_ - x3_)

    #@code_unrolled Base.copyto!(g, x, y) # FIXME not working
    bmk = @benchmark copyto!($g, $x, $y)
    display(bmk)
    @test bmk.allocs == 0

    display(g)

    # get(graph,tuple)
    @time y_ = get(g, x)

    (y1_, y2_, y3_) = y_
    @test all(y1_ == x1_ + x2_)
    @test all(y2_ == x1_ - x2_)
    @test all(y3_ == x2_ - x3_)

    bmk = @benchmark get($g, $x)
    display(bmk)
    @test bmk.allocs <= 11  # FIXME 
end
