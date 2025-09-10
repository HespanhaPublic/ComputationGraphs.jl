# Representation of computation graphs

This section of the manual documents the inner workings of the graph computation functions in the
source file
[src/compute.jl](https://github.com/hespanha/ComputationGraphs/blob/main/src/Compute.jl).

!!! warning
    The timing information reported here is obtained by running `Documenter`'s `@example` triggered by a
    Github action. Because of this, there is no control over the hardware used and consequently the
    timing values that appear in this document are not very reliable; especially in what regards
    [Parallel computations](@ref).

## Computation nodes

An expression like `A*x+b` contains

+ three nodes `A`, `b`, and `x` that corresponds to variables, and
+ two computation nodes, one for the multiplication and the other for the addition.

Since we are aiming for allocation-free computation, we start by pre-allocating memory for all nodes

```@example representation1
using Random
begin #hide
A = rand(Float64,400,30)  # pre-allocated storage for the variable A
x = rand(Float64,30)     # pre-allocated storage for the variable b
b = rand(Float64,400)     # pre-allocated storage for the variable x
Ax = similar(b)          # pre-allocated storage for the computation node A*x
Axb = similar(b)         # pre-allocated storage for the computation node A*x+b
nothing # hide
end # hide
```

and associate to the following functions to the two computation nodes:

```@example representation1
using LinearAlgebra
begin # hide
function node_Ax!(out::Vector{F},in1::Matrix{F}, in2::Vector{F}) where {F} 
    mul!(out,in1,in2)
end
function node_Axb!(out::Vector{F},in1::Vector{F}, in2::Vector{F}) where {F} 
    @. out = in1 + in2
end
nothing # hide
end # hide
```

It would be temping to construct the computation graph out of such functions. However, every
function in julia has is own unique type (all subtypes of the `Function` abstract type). This is
problematic because we will often need to iterate over the nodes of a graph, e.g., to re-evaluate
all nodes in the graph or just the parents of a specific node. If all nodes have a unique type, then
such iterations not be type-stable.

To resolve this issue we do two "semantic" transformations to the functions above: [function
closure](https://docs.julialang.org/en/v1/devdocs/functions/#Closures) and function wrapping with
the package [FunctionWrappers](https://github.com/JuliaLang/FunctionWrappers.jl).

### Function closure

[Function closure](https://docs.julialang.org/en/v1/devdocs/functions/#Closures) allow us to
obtain functions for all the nodes that "look the same" in the following sense:

+ they all have they have the same signature (i.e., same number of input parameters and with the
    same types), and
+ they all return a value of the same type.

Specifically, we "capture" the input parameters for the two computation nodes, which makes them look
like parameter-free functions that return nothing:

```@example representation1
begin # hide
@inline node_Ax_closed!() = let  Ax=Ax , A=A, x=x
    node_Ax!(Ax,A,x)
    nothing
    end
@inline node_Axb_closed!() = let Axb=Axb, Ax=Ax, b=b
    node_Axb!(Axb,Ax,b)
    nothing
    end
nothing # hide
end # hide
```

!!! note
    See [Performance tips on the performance of captured
    variables](https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured) on
    the use of `let`, which essentially helps the parser by "fixing" the captured variable. To be precise
    "fixing" the arrays, but not the values of their entries.

### Function wrapping

Even though all node functions now have similar inputs and outputs, they are still not of the same
type (as far as julia is concerned). To fix this issue, we use the package
[FunctionWrappers](https://github.com/JuliaLang/FunctionWrappers.jl) to create a type-stable
wrapper:

```@example representation1
begin # hide
import ComputationGraphs
node_Ax_wrapped = ComputationGraphs.FunctionWrapper(node_Ax_closed!)
node_Axb_wrapped = ComputationGraphs.FunctionWrapper(node_Axb_closed!)
nothing # hide
end # hide
```

The "wrapped" functions can be called directly with:

```julia
begin # hide
node_Ax_wrapped()
node_Axb_wrapped()
nothing # hide
end # hide
```

or a little faster with

```@example representation1
begin # hide
ComputationGraphs.do_ccall(node_Ax_wrapped)
ComputationGraphs.do_ccall(node_Axb_wrapped)
nothing # hide
end # hide
```

!!! warning
    The code above does not actually use
    [FunctionWrappers](https://github.com/JuliaLang/FunctionWrappers.jl); instead it uses a very simplified version of [FunctionWrappers](https://github.com/JuliaLang/FunctionWrappers.jl) that can only wrap functions with no arguments that always return `nothing`.

    To use [FunctionWrappers](https://github.com/JuliaLang/FunctionWrappers.jl), we would have used instead

    ```julia
    import FunctionWrappers
    node_Ax_wrapped_FW = FunctionWrappers.FunctionWrapper{Nothing,Tuple{}}(node_Ax_closed!)
    node_Axb_wrapped_FW = FunctionWrappers.FunctionWrapper{Nothing,Tuple{}}(node_Axb_closed!)
    ```

    and the functions would be called with

    ```julia
    begin # hide
    FunctionWrappers.do_ccall(node_Ax_wrapped_FW, ())
    FunctionWrappers.do_ccall(node_Axb_wrapped_FW, ())
    nothing # hide
    end # hide
    ```

### Verification

We can now check the fruits of our work.

+ Type stability?

```@example representation1
begin # hide
println("Type stability for original: ", typeof(node_Ax!)==typeof(node_Axb!))
println("Type stability for wrapped : ", typeof(node_Ax_wrapped)==typeof(node_Axb_wrapped))
nothing # hide
end #hide
```

+ Correctness?

```@example representation1
begin # hide
rand!(A)
rand!(b)
rand!(x)

# the original functions
node_Ax!(Ax,A,x)
node_Axb!(Axb,Ax,b)
println("Correctness for original: ", Axb==(A*x+b))

rand!(A)
rand!(b)
rand!(x)

# the new functions
ComputationGraphs.do_ccall(node_Ax_wrapped)
ComputationGraphs.do_ccall(node_Axb_wrapped)
println("Correctness for wrapped : ", Axb==(A*x+b))
nothing # hide
end # hide
```

+ Speed?

```@example representation1
using BenchmarkTools, Printf
begin # hide
@show Threads.nthreads()
BLAS.set_num_threads(1)
@show BLAS.get_num_threads()
@show Base.JLOptions().opt_level

bmk1 = @benchmark begin
    node_Ax!($Ax,$A,$x)
    node_Axb!($Axb,$Ax,$b)
end evals=1000 samples=10000
println("Original:\n",sprint(show,"text/plain",bmk1;context=:color=>true)) # hide

bmk3 = @benchmark begin
    node_Ax_closed!()
    node_Axb_closed!()
end evals=1000 samples=10000
println("Closure:\n",sprint(show,"text/plain",bmk3;context=:color=>true)) # hide

bmk2 = @benchmark begin
    ComputationGraphs.do_ccall($node_Ax_wrapped)
    ComputationGraphs.do_ccall($node_Axb_wrapped)
end evals=1000 samples=10000
println("Wrapped:\n",sprint(show,"text/plain",bmk2;context=:color=>true)) # hide

@printf("Overhead due to closure  = %3.f ns\n",median(bmk3.times)-median(bmk1.times))
@printf("Overhead due to wrapping = %3.f ns\n",median(bmk2.times)-median(bmk3.times))
@printf("Total overhead           = %3.f ns\n",median(bmk2.times)-median(bmk1.times))
nothing # hide
end # hide
```

This shows that closure and wrapping do introduce a small overhead (tens of ns). However, the
benefits of type stability will appear when we start iterating over nodes. To see this consider the
following function that evaluates a set of nodes:

```@example representation1
begin # hide
function compute_all!(nodes::Vector{Function})
    for node in nodes
        node()
    end
end
function compute_all_wrapped!(nodes::Vector{ComputationGraphs.FunctionWrapper})
    for node::ComputationGraphs.FunctionWrapper in nodes
        ComputationGraphs.do_ccall(node)
    end
end
nothing # hide
end # hide
```

We can use `@code_warntype` to see how wrapping helps in terms of type stability:

```@example representation1
using InteractiveUtils # hide
begin # hide
# using just closure
nodes_closed=repeat([node_Ax_closed!,node_Axb_closed!],outer=5)
@show typeof(nodes_closed)
InteractiveUtils.@code_warntype compute_all!(nodes_closed)
#println(sprint(code_warntype,compute_all!,(typeof(nodes_closed),);context=:color=>true)) # hide

# using closure+wrapped
nodes_wrapped=repeat([node_Ax_wrapped,node_Axb_wrapped],outer=5)
@show typeof(nodes_wrapped)
InteractiveUtils.@code_warntype compute_all_wrapped!(nodes_wrapped)
#println(sprint(code_warntype,compute_all_wrapped!,(typeof(nodes_wrapped),);context=:color=>true)) # hide
nothing # hide
end # hide
```

These specific functions `compute_all!` and `compute_all_wrapped!` are so simple that type
instability actually does not lead to heap allocations, but the use of wrapped functions still leads
to slightly faster code.

```@example representation1
begin # hide
@show typeof(nodes_closed)
bmk3 = @benchmark compute_all!($nodes_closed) evals=1 samples=10000
println("Closure:\n",sprint(show,"text/plain",bmk3;context=:color=>true)) # hide

@show typeof(nodes_wrapped)
bmk2 = @benchmark compute_all_wrapped!($nodes_wrapped)  evals=1 samples=10000
println("Closure+Wrapping:\n",sprint(show,"text/plain",bmk2;context=:color=>true)) # hide

nothing # hide
end # hide
```

## Conditional computations

So far we discussed how to compute *all nodes* or some *give vector of nodes*. Restricting
evaluations to just the set of nodes that *need* to be recomputed requires introducing some
simple logic to the function closures.

### Implementation

To support need-based evaluations, we use a `BitVector` to keep track of which nodes have been
evaluated. For our 2-node example, we would use

```@example representation1
validValue=falses(2)
nothing # hide
```

The functions below now include the logic for need-based evaluation:

```@example representation1
begin # hide
node_Ax_conditional_closed() = let validValue=validValue, 
    Ax=Ax , A=A, x=x
    node_Ax!(Ax,A,x)    # this node's computation
    nothing
    end
node_Ax_conditional_wrapped = ComputationGraphs.FunctionWrapper(node_Ax_conditional_closed)
node_Axb_conditional_closed() = let validValue=validValue, 
    Axb=Axb, Ax=Ax, b=b, 
    node_Ax_conditional_wrapped=node_Ax_conditional_wrapped
    # compute parent node Ax (if needed)
    if !validValue[1]
        validValue[1]=true
         ComputationGraphs.do_ccall(node_Ax_conditional_wrapped)
    end
    node_Axb!(Axb,Ax,b)  # this nodes' computation
    nothing
    end
node_Axb_conditional_wrapped = ComputationGraphs.FunctionWrapper(node_Axb_conditional_closed)
nothing # hide
end # hide
```

With this logic, we only need a call to evaluate the node `A*x+b`, as this will automatically
trigger the evaluation of `A*x` (if needed). To check that the logic is working, we do:

```julia
begin # hide
fill!(validValue,false)
fill!(Ax,0.0)
fill!(Axb,0.0)
ComputationGraphs.do_ccall(node_Ax_conditional_wrapped)
@assert validValue == [false,false] "no parent computed"
@assert all(Ax .== A*x)  "should only compute Ax"
@assert all(Axb .== 0) "should only compute Ax"

fill!(validValue,false)
fill!(Ax,0.0)
fill!(Axb,0.0)
ComputationGraphs.do_ccall(node_Axb_conditional_wrapped)
@assert validValue == [true,false] "parent should have been computed"
@assert all(Ax .== A*x)  "should compute both"
@assert all(Axb .== A*x+b) "should compute both"
nothing # hide
end # hide
```

### Timing verification

We can now check the impact of the new logic on timing.

```@example representation1
begin # hide
using BenchmarkTools, Printf
@show Threads.nthreads()
BLAS.set_num_threads(1)
@show BLAS.get_num_threads()
@show Base.JLOptions().opt_level

bmk1 = @benchmark begin
    node_Ax!($Ax,$A,$x)
    node_Axb!($Axb,$Ax,$b)
end evals=1000 samples=10000
println("Unconditional computation:\n",sprint(show,"text/plain",bmk1;context=:color=>true)) # hide

bmk2a = @benchmark begin
    ComputationGraphs.do_ccall($node_Ax_wrapped)
    ComputationGraphs.do_ccall($node_Axb_wrapped)
end evals=1000 samples=10000
println("Unconditional computation with wrapping:\n",sprint(show,"text/plain",bmk2;context=:color=>true)) # hide

bmk2b = @benchmark begin
    $validValue[1]=false
    $validValue[2]=false
    if !$validValue[2]
        $validValue[2]=true
        ComputationGraphs.do_ccall($node_Axb_conditional_wrapped)
    end
end evals=1000 samples=10000
println("Conditional computation, but with all valid=false:\n",sprint(show,"text/plain",bmk2;context=:color=>true)) # hide

bmk3 = @benchmark begin
    if !$validValue[2]
        $validValue[2]=true
        ComputationGraphs.do_ccall($node_Axb_conditional_wrapped)
    end
end evals=1 samples=10000
println("Conditional computation, with full reuse:\n",sprint(show,"text/plain",bmk3;context=:color=>true)) # hide

bmk4 = @benchmark begin
    $validValue[2]=false
    if !$validValue[2]
        $validValue[2]=true
        ComputationGraphs.do_ccall($node_Axb_conditional_wrapped)
    end
end evals=1000 samples=10000
println("Conditional computation, with valid=false only for Axb:\n",sprint(show,"text/plain",bmk4;context=:color=>true)) # hide

@printf("overhead due to closure+wrapping for full computations          = %+6.f ns\n",
    median(bmk2a.times)-median(bmk1.times))
@printf("overhead due to closure+wrapping+logic for full computations    = %+6.f ns\n",
    median(bmk2b.times)-median(bmk1.times))
# @printf("overhead due just to             logic for full computations    = %+6.f ns\n", # hide 
#      median(bmk2b.times)-median(bmk2a.times)) # hide
@printf("overhead due to closure+wrapping+logic for for computations     = %+6.f ns (<0 means savings)\n",
    median(bmk3.times)-median(bmk1.times))
@printf("overhead due to closure+wrapping+logic for partial computations = %+6.f ns (<0 means savings)\n",
    median(bmk4.times)-median(bmk1.times))
nothing # hide
end # hide
```

As expected, much time is saved when re-evaluations are not needed. When they are needed, the
logic adds a small additional penalty.

!!! note
    The code above is the basis for [ComputationGraphs.generateComputeFunctions](@ref).

## Parallel computations

Parallel evaluation are implemented by associating to each computation node one `Threads.Task` and one pair of
`Threads.Events`. For each computation node `i`:

+ The task `task[i]::Threads.Task` is responsible carrying out the evaluation of node `i` and
  synchronizing it with the other nodes.
+ The event `request[i]::Threads.Event(autoreset=true)` is used to request `task[i]` to evaluate its
  node, by issuing `notify(request[i])`.
+ The event `valid[i]::Threads.Event(autoreset=false)` is used by node `i` to notify all other nodes
  that it has finished handling a computation request received through `request[i]`

The following protocol is used:

+ All node tasks are spawn simultaneously and each task `i` immediately waits on `request[i]` for evaluation request.

+ Upon receiving a request, task `i` checks which of its parents have valid data:

  1) For every parent `p` with missing data, it issues an evaluation request using `notify(request[p])`.
  2) After that, the task waits on the requests to be fulfilled by using `wait(valid[p])` for the
     same set of parent node.

+ Once all parents have valid data, node `i` performs its own computation and notifies any waiting
  child node that its data became valid using `notify[valid[i]]`.

The operation described above makes the following assumptions:

+ Any thread that needs the value of node `i` should first issues an evaluation request
  using `notify(request[i])` and then wait for its completion using `wait(valid[i])`.

+ When the value of a variable `v` changes, all its children nodes `c` need to be notified that their
  values become invalid by issuing `reset(valid[c])`.
  
+ To avoid races, these last `reset(valid[c])` *cannot* be done while computations are being performed.

!!! warning
    The last assumption above should be enforced by an explicit locking mechanism, but that has not yet been implemented.

!!! warning
    For very large matrix multiplications, BLAS makes good use of multiple threads. In this
    case, we should not expect significant improvements with respect to evaluating the computation
    graph sequentially. Instead, it is better to allow BLAS to manage all the threads, with a
    sequential evaluation of the computation graph.

### Parallelism implementation

We will illustrate the mechanism above with the computation of `A*x+B*y` for which the two
multiplications can be parallelized. The corresponding graph has

+ three nodes `A`,`x`,`B`,`y` that corresponds to variables; and
+ three computation nodes, two for each of the multiplications and the other for the addition.

We start by pre-allocating memory for all nodes

```@example representation2
using Random
begin #hide
A = rand(Float64,4000,3000)
x = rand(Float64,3000,1000) 
B = rand(Float64,4000,2500)
y = rand(Float64,2500,1000) 
Ax = Matrix{Float64}(undef,4000,1000)
By = similar(Ax)
AxBy = similar(Ax)
nothing # hide
end # hide
```

and defining the computations for each node

```@example representation2
using LinearAlgebra
begin # hide
function node_Ax!(out::Matrix{F},in1::Matrix{F}, in2::Matrix{F}) where {F} 
    mul!(out,in1,in2)
end
function node_By!(out::Matrix{F},in1::Matrix{F}, in2::Matrix{F}) where {F} 
    mul!(out,in1,in2)
end
function node_AxBy!(out::Matrix{F},in1::Matrix{F}, in2::Matrix{F}) where {F} 
    @. out = in1 + in2
end
nothing # hide
end # hide
```

We used fairly big matrices for which the computation takes some time, as we can see below:

```@example representation2
begin # hide
@show Threads.nthreads()
BLAS.set_num_threads(1)
@show BLAS.get_num_threads()
@show Base.JLOptions().opt_level

node_Ax!(Ax,A,x) # compile # hide
node_By!(By,B,y) # compile # hide
node_AxBy!(AxBy,Ax,By) # compile # hide
@time begin
    node_Ax!(Ax,A,x)
    node_By!(By,B,y)
    node_AxBy!(AxBy,Ax,By)
end
nothing # hide
end # hide
```

To implement the parallelization mechanism described above we need 2 event-triggered objects per
node:

```@example representation2
begin # hide
valid=Tuple(Threads.Event(false) for _ in 1:3)
request=Tuple(Threads.Event(true) for _ in 1:3)
reset.(valid)
reset.(request)
nothing # hide
end # hide
```

The computation tasks can then be launched using

```@example representation2
begin # hide
tasks=[
    Threads. @spawn while true
        wait(request[1])
        if !valid[1].set
            node_Ax!(Ax,A,x)    # this node's computation
            notify(valid[1])
        end
    end

    Threads. @spawn while true
        wait(request[2])
        if !valid[2].set
            node_By!(By,B,y)    # this node's computation
            notify(valid[2])
        end
    end

    Threads.@spawn while true
        wait(request[3])
        if !valid[3].set
            valid[1].set || notify(request[1])
            valid[2].set || notify(request[2])
            valid[1].set || wait(valid[1])
            valid[2].set || wait(valid[2])
            node_AxBy!(AxBy,Ax,By)  # this node's computation
            notify(valid[3])
        end
    end
]
println(sprint(show,"text/plain",tasks)) # hide
nothing # hide
end # hide
```

!!! note
    Very similar code is used in [computeSpawn!](@ref) to parallelize the computation of general graphs.

### Parallelism verification

To verify the operation of the approaches outlined above, we make a request for the value of the
final node `A*x+B*y` and wait on the node being valid:

```@example representation2
begin # hide
using ThreadPinning
pinthreads(:cores)
@show Threads.nthreads()
BLAS.set_num_threads(1)
@show BLAS.get_num_threads()
@show Base.JLOptions().opt_level

fill!(Ax,0.0)
fill!(By,0.0)
fill!(AxBy,0.0)
begin # compile # hide
    notify(request[3]) # compile # hide
    wait(valid[3]) # compile # hide
end # compile # hide
reset.(valid)
println("valid before :",getproperty.(valid,:set))
@time begin
    notify(request[3])
    wait(valid[3])
end
println("valid after  :",getproperty.(valid,:set))
@assert Ax==A*x 
@assert By==B*y
@assert AxBy==A*x+B*y
nothing # hide
end # hide
```

When multiple hardware threads are available, the time reported by `@time` is roughly about half,
showing a good use of the threads.

!!! note
    We can see whether the julia threads were successfully "pinned" to physical hardware threads using
    `ThreadPinning.threadinfo()`, where *red* means that multiple julia threads are running on the
    same hardware thread and *purple* means that the julia thread is really running on a hyperthread. In
    either case, we should not expect true parallelism. This is often the case when code is run through
    a GitHub action (as in generating this manual page) on a computer with a single core with
    Simultaneous Multithreading (SMT).

    ```@example representation2
    begin # hide
    using ThreadPinning
    @show Threads.nthreads()
    @show pinthreads(:cores)
    threadinfo()
    nothing # hide
    end # hide
    ```
