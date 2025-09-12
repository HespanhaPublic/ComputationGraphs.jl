
# Symbolic differentiation

## Differentiation

`ComputationGraphs` provides basic tools for *symbolics differentiation*. Not to be confused with
*automatic differentiation*, which evaluates partial derivatives of a function as the function is
being evaluated.

`ComputationGraphs` computes derivatives *symbolically* in three steps:

1) It operates on expressions encoded in the computation graph;
2) computes the derivative of one node of the graph symbolically with respect to a given variable
   (using the standard calculus rules for differentiation); and
3) then dds the symbolic expression of the derivative to the computation graph.  

Two points need emphasis:

+ The symbolic differentiation is performed *when the graph is being built*, rather than while the
  expressions are being evaluated as it is typically done in automatic differentiation.
  
  As we shall see in [Algebraic simplifications](@ref), this permits the discovery of
  "simplification" that can speed up computations.

+ Typically, the derivative will reuse existing graph nodes (as we saw in the example in [Building a
  computation graph](@ref)). This also enables significant computational savings by reusing
  computation across the evaluation of a function and its derivative.

The function [D](@ref) is used to compute the partial derivative of an expression with respect to
a variable; or more precisely, to augment the graph with the formula that computes the partial
derivative.

The computation graph in [Building a computation graph](@ref) could have been generated
using [D](@ref) as follows:

```julia
using ComputationGraphs
graph = ComputationGraph(Float64)
A = variable(graph, 4, 3)
x = variable(graph, 3)
b = variable(graph, 4)
e = @add graph norm2(A*x-b)
grad = D(graph,e,x)
```

!!! note
    The graph generated using [D](@ref) actually has a few extra nodes, but these nodes never
    really need to be computed due to [Algebraic simplifications](@ref).

!!! warning
    Currently, `ComputationGraphs` only includes a relatively small set of rules for symbolic
    differentiation. These are expected to grow and the package matures.

    The set of rules currently used can be found in [symbolic.pdf](../symbolics.pdf)

## Algebraic simplifications

`ComputationGraphs` include a small set of (very simple) rules that automatically simplify a
computation graph at the time it is being creates. These include

+ The sum of any expression with a scalar/vector/matrix zero, does not change the value of the expression.
+ Any product with a zero scalar/vector/matrix is always zero.
+ The adjoint of an adjoint returns the original expression.
+ Multiplication of a matrix by a vector of ones, corresponds to replacing each row of the matrix by
  the sum of its elements.
+ Multiplication of a matrix by the k-th vector of a canonical basis, extracts the k-th column of
  the matrix.
+ Etc.

These rules are applied *when the graph is being built* to "reduce" the graph. For example, if we
try to build a graph for the expression `y = 0 * a + b`, we actually end up with the graph of
`y = b`. To be precise, the graph will actually have nodes for `0` and `a`, but `y` is obtained
directly from `b` without any multiplication and addition.

It might seem that such rules are too simple to be useful, as no one would ever try to encode into a
computation graph the expression `0 * a`. However, such computations actually arise very often in
differentiation. For example, when applying the product rule to compute the derivate of `a * x + b`
with respect to `x`, we get

```math
\begin{align*}
\nabla_x (a x + b) &= a (\nabla_x x) + (\nabla_x a) x + \nabla_x b\\
&=a \times 1 + 0 \times x + 0\\
&=a
\end{align*}
```

where the last equality results precisely from applying the very simple rules listed above. Because
these rules are applied *when the graph is being built* it is immediately discovered that evaluating
this partial derivative actually does not require any computation.

!!! warning
    Currently, `ComputationGraphs` only includes a very small set of rules for symbolic simplification. These are expected to grow and the package matures.

    When trying to different an expression not supported by the current set of rules a `DomainError` exception is thrown.
