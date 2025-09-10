# Basics

`ComputationGraphs` is about improving the speed (and energy consumption) of numerical computations
that need to be performed repeatedly, e.g.,
+ one iteration of a numerical optimization algorithm, 
+ one iteration of a filtering/smoothing algorithm,
+ repeated calls to a classification algorithm on different samples, etc.

The computation to be performed is encoded into a data structure [ComputationGraph](@ref) that
permits several forms of "run-time" optimization, including:
+ allocation-free operation
+ (partial) re-use of perviously performed computations
+ symbolic differentiation
+ symbolic algebraic simplifications 

## What is a *computation graph*?

A *computation graph* represents a set of mathematical operations, 
 performed over a set of variables. Formally, it is represented as a *graph* with:

+ *nodes* that correspond to variables that hold the result of a mathematical operation, and 

+ *edges* that encode the *operands* used by each mathematical operation.

The edges are directed from the operand (which we call the "parent node") to the result of the operation
(which we call the "child node")


For example, the formula

```math
    e = \| A\, x -b \|^2
```

can be represented by the following tree

```
norm2(A*x-b) (operation is squared-norm \| \|^2)
│
A*x-b        (operation is subtraction -)
│
├── A*x      (operation is multiplication *)
│   ├── A
│   └── x
└── b
```

We can recognize two types of nodes:
1) `A`, `x`, and `x` are *varible nodes* that are associated with "inputs" to the computation, and
2) `A*x`, `A*x-b`, and `norm2(A*x-b)` are *computation nodes* that are associated with some
   algebraic operation.

We will often use the expression "evaluate a (computation) node" to mean "perform the operation
associated with a node". For example, by "evaluate the node `A*x-b`" we mean:

1) first fetch the value of the parent node `A*x`,
2) then fetch the value of the other parent node ``b`,
3) and finally subtract these two values to obtain the value of the child node `A*x-b`.

Since step 1) involves a computation node, this step presumes that the parent node `A*x` has been
previously evaluated and the value of this evaluation has been saved; otherwise this node would need
to be evaluated, prior to step 1).

Computation graphs are represented by the structures [ComputationGraph](@ref), which encode the
graph itself as well as additional information about the variables (size, types, whether the node
has been evaluated, its stored value, etc).

!!! note

    While missing from this example, a third type of node is possible: *constant* nodes are also associated with "inputs" to the computation (like *variable* nodes), but they never change. Declaring "input" nodes as constants typically enables computational savings.

## Building a computation graph

The computation graph above can be created using the following code

```@example guide1
using ComputationGraphs
graph = ComputationGraph{Float64}()
A = variable(graph, 4, 3)
x = variable(graph, 3)
b = variable(graph, 4)
e = @add graph norm2(A*x-b)
nothing # hide
```

Upon execution `graph` represents a 6-node computation graph:
1) The 1st assignment creates an empty graph
2) The 2nd assignment creates a variable that stores a 4x3 matrix
3) The 3rd and 4th assignments create 2 variables that store vectors with sizes 3 and 4, respectively.
4) The 5th assignment adds nodes to the graph that store the values of `A*x`, `A*X-b`, and `norm2(A*X-b)`.

This code only *defines* a computation graph, but it actually does not performed any computation.

A computation graph may include several computations that share common variables. For example, we
could add to the same graph the computation of the gradient of `e` with respect to `x`, which turns
out to be

```math
\nabla_x e = 2A'(A\,x-b)
```

This can be added to the existing computation graph using

```@example guide1
grad = @add graph constant(2.0) * adjointTimes(A, A*x - b )
nothing # hide
```

This command will recognize that the existing graph already nodes for `A*x` and `A*x-b` so only 3
more nodes need to be added: the constant value `2` and the products `A'*(A*x-b)`, `2*A'*(A*x-b)`;
resulting in a graph with 9 nodes.

The reuse of nodes has important implications in terms of reusing computation. Specifically, 
1) Once we compute the gradient $\nabla_x e$, the term `A*x-b` becomes available and computing `e`
   only requires computing [norm2](@ref) of `A*x-b`, which is a relatively "cheap" computation.
2) Alternatively, if we first compute `e`, then  `A*x-b` becomes available and computing the
   gradient only requires multiplying it by `2*A`.
In either case, we can share intermediate results between the computations of `e` and $\nabla_x e$. 

!!! note

    We used `adjointTimes(M,N)` to represent the operation `M'*N`, which really consists of two operations: taking the adjoint/transpose of the first matrix and then multiplying it by the second matrix.
 
    In practice, it is generally more efficient (in terms of time and memory) to combine the two operations into a single one, which is generally automatically done by `LinearAlgebra`. 

    We currently force the user to explicitly decide whether or not to combine the two operations by using

      + `adjointTimes(M,n)` -- combine, or
      + `adjoint(M)*N` -- do not combine.
 
    The later option can be better if `adjoint(M)` will turn out to be useful for other computations.

!!! warning 

    Currently, `ComputationGraphs` only supports a relatively small set of algebraic operations. These are expected to grow and the package matures.

## Using a computation graph

The creation of a computation graph encodes relationships between variables, but does not actually
perform any computation. To perform computations we need to:

+ first set the values of all variables that appear in the graph, and
+ second carryout the computations (in the appropriate order)

The first step uses the [set!](@ref) command to set the values of variable:

```@example guide1
set!(graph, A, [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0; 10.0 11.0 12.0])
set!(graph, x, [1.0, 1.0, 1.0])
set!(graph, b, [2.0, 2.0, 2.0, 2.0])
nothing # hide
```

We are now ready to perform the graph computations using [compute!](@ref), which can
take two forms: To recompute the whole graph, we can use

```@example guide1
compute!(graph)
nothing # hide
```

but if we just want to recompute the portion of the graph needed to evaluate the node `e`, we can use

```@example guide1
compute!(graph, e)
nothing # hide
```

Finally, we can get the value of `e` using [get](@ref)

```@example guide1
value = get(graph,e)
println(value)
nothing # hide
```

!!! note
    The values of all nodes in a computation graph are stored as N-dimensional arrays. Vectors are 1-dimensional arrays, matrices 2-dimensional arrays, but higher dimensional arrays can also be used in ComputationGraphs
    
    Scalars turn out to also be represented as arrays, but 0-dimensional arrays, which in julia always have a single element. 0-dimensional arrays can be created using `fill(value)` and are displayed as `fill(value)`:

    ```@example
    x = fill(1.0)
    println(x)
    nothing # hide
    ```

## Reusing computations

The structure [ComputationGraph](@ref) encodes all the dependencies between the nodes of a
computation graph, which enables minimizing computation by maximizing the re-use of computations
that have been previously performed.

### Doing all the *necessary* computations, but no more than that

The goal of the `compute!(graph)` is to make sure that all nodes hold *valid value*. This does not
mean that the function needs to recompute all nodes, since some nodes may have been previously
compute and thus may already hold valid values.

This means that if we call `compute!(graph)` twice, the second time will actually not perform any
computation. This can be seen in the following example that "fools" `@benchmark` into believing that
the operation $$\|Ax-b\|$$ for very large matrices/vectors only takes a few nano seconds.

```@example
using ComputationGraphs, BenchmarkTools
graph = ComputationGraph{Float64}()
A = variable(graph, rand(Float64,4000, 3000))
x = variable(graph, rand(Float64,3000))
b = variable(graph, rand(Float64,4000))
e = @add graph norm2(A*x-b)
using BenchmarkTools
bmk=@benchmark compute!($graph)
println(sprint(show,"text/plain",bmk;context=:color=>true));nothing # hide
```

To prevent computation re-use, we can use the keyword `force=true` to force [compute!](@ref) to
actually redo the computations:

```@example
using ComputationGraphs, BenchmarkTools
graph = ComputationGraph{Float64}()
A = variable(graph, rand(Float64,4000, 3000))
x = variable(graph, rand(Float64,3000))
b = variable(graph, rand(Float64,4000))
e = @add graph norm2(A*x-b)
using BenchmarkTools
bmk=@benchmark compute!($graph,force=true)
println(sprint(show,"text/plain",bmk;context=:color=>true));nothing  # hide
```

We now see that the computation actually takes a few milliseconds.

### Partial-graph computations

As noted above, the function [compute!](@ref) can be made node-specific; meaning that it only
recomputes the set of nodes that are "need" to get the value of a specific node. Also in this case,
"parent nodes" that already hold *valid values* do not need to be recomputed.

The function [get](@ref) also checks if the desired node has a *valid value* and recomputes the
node if it does not. Specifically, `get(graph,node)` implicitly calls `compute(graph,node)` before
returning the value of the node. This means that in the following examples the 2nd function
will actually never do any computation:

```julia
compute!(graph)
compute!(graph, e)   # no computation
```

```julia
compute!(graph, e)
value=get(graph, e)  # no computation, just returns value
```

```julia
value=get(graph, e)
compute!(graph, e)   # no computation
```

### Redoing computations when *necessary*

Initialization of updates in variables are prompted by calling either of the following two functions:

+ `set!(graph, variable, value)` sets the value of the node `variable` (which must have been created
  using the command [variable](@ref)) to the value of the array `value`.

+ `coptyto!(graph, node1, node2)` copies the value of the node `node2` to the node `variable`
  (which must have been created using the command [variable](@ref)).

When the values of input variables are changed, some (but not necessarily all) nodes may need to be
recomputed. 

+ For example, if we use [set!](@ref) or [copyto!](@ref) to change the values of the three variables
  `A`, `x`, `b`, their children nodes `A*x`, `A*x-b`, and `norm2(A*x-b)` need to be recomputed.

+ However, if we only use [set!](@ref) or [copyto!](@ref) to change the value of `b`, the node `A*x`
  does not need to be recomputed.

The functions [set!](@ref) and [copyto!](@ref) are "smart" in the sense that they keeps track of
these dependencies and only marks as "invalid" the "children" of the variable that has been changes.
The following examples illustrate this: In the first case, the value of `x` is set so recomputing
`e` requires recomputing the "expensive" product `A*x`:

```@example guide2
using ComputationGraphs, BenchmarkTools
graph = ComputationGraph{Float64}()
A = variable(graph, rand(Float64,4000, 3000))
x = variable(graph, rand(Float64,3000))
b = variable(graph, rand(Float64,4000))
e = @add graph norm2(A*x-b)
x0=rand(Float64,size(x))
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5 # hide
bmk=@benchmark begin
        set!($graph, $x, $x0)   # x changes so A*x and A*x-b need to be recomputed
        value=get($graph, $e)
    end
println(sprint(show,"text/plain",bmk;context=:color=>true)); nothing # hide
```

In this example, only the value of `b` is set so recomputing `e` can reuse the previous value of
`A*x`. Subtractring the new `b` and taking the norm are much "cheaper" operations ans the compute
time decreases from milliseconds to microseconds: 

```@example guide2
b0=rand(Float64,size(b))
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5 # hide
bmk=@benchmark begin
        set!($graph, $b, $b0)   # b changes so A*x does not need to be recomputed
        value=get($graph, $e)
    end
println(sprint(show,"text/plain",bmk;context=:color=>true));nothing # hide
```

### Allocation-free computations

The sizes of the arrays associated with all nodes become known as the graph is built, which means
that memory can be pre-allocated to store the values associated with every node. 

In practice, this means that calls to [set!](@ref), [copyto!](@ref), [get](@ref), and
[compute!](@ref) are typically allocation free, as reported above by `@benchmark`. This greatly
contributes to minimizing garbage collection and keeping the compute times small and fairly
predictable.

