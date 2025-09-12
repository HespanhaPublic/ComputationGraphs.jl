export gradDescent!, adam!

"""
    (;next_theta,eta,gradients) = gradDesc(graph; loss, theta)

Recipe used to performs the computations needed by the classical gradient descent algorithm to
minimize a (scalar-valued) *loss function* \$J(\theta)\$ by adjusting a set of *optimization
parameters* \$\\theta\$, according to

```math
    \\theta^+ = \\theta - \\eta\\, \\nabla_\\theta J(\\theta)
```

# Parameters:

+ `graph::ComputationGraph`; Computation graph that is updated "in-place" by adding to it
        all the nodes needed to perform one step of gradient descent.

+ `loss::Node`: Scalar-valued computation node that corresponds to the loss function \$J(\\theta)\$

+ theta::NamedTuple`: Named tuple with the variable nodes that correspond to the optimization
        parameters \$\\theta\\\$.

# Returns: named tuple with

+ `eta::Node`: Scalar-valued variable node that can be used to set the *learning rate* \$\\eta\$.

+ `next_theta::Tuple`: Named tuple with the computation nodes that holds the value \$\\theta^+\$ of
        the optimization parameters *after one gradient descent* iteration

+ `gradients::NamedTuple`: Named tuple of the computation nodes that hold the value of the gradients of
        the loss function with respect to the different variables in `theta`.

# Example:

```julia
using ComputationGraphs

graph = ComputationGraph(Float64)

# Define optimization parameters and loss function
A = variable(graph, 4, 3)
x = variable(graph, 3)
b = variable(graph, 4)
loss = @add graph norm2(times(A, x) - b)

# Call gradDescent! recipe
theta = (;x,)
(; next_theta, eta, gradients) = gradDescent!(graph; loss, theta)

# Set fixed parameters
set!(graph, A, [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0; 10.0 11.0 12.0])
set!(graph, b, [2.0, 2.0, 2.0, 2.0])

# Set learning rate
set!(graph, eta, 0.001)

# Initialize optimization parameter
set!(graph, x, [1.0, 1.0, 1.0])
println("initial loss: ", get(graph,loss))

# Gradient descent loop
for i in 1:100
    compute!(graph, next_theta)       # compute next value of theta
    copyto!(graph, theta, next_theta) # execute update
end
println("final loss: ", get(graph,loss))
```
"""
function gradDescent!(
    graph::ComputationGraph;
    loss::Node,
    theta::NamedTuple,
    code::Union{Code,Nothing}=nothing,
) where {Node<:AbstractNode}
    # create solver parameters
    eta = variable(graph, ())
    # compute gradients
    gradients = (; (
        k => D(graph, loss, v) for (k, v) in pairs(theta)
    )...)
    # next_theta = theta - eta * grad
    next_theta = (; (
        k => -(graph,
            getindex(theta, k),
            scalarTimes(graph, eta, grad))
        for (k, grad) in pairs(gradients)
    )...)

    if !isnothing(code)
        ## Add code
        sets!(code, eta => "setGradDescent!_eta!")
        gets!(code, (; next_moments..., next_theta...) => "getNext")
        copies!(code, (next_theta => theta) => "update!")
    end


    return (;
        eta,
        next_theta,
        gradients)
end

"""
    (;  eta, beta1, beta2, epsilon,
        init_state, state, next_state,
        next_theta, gradients) = adam!(graph; loss, theta)

Recipe used to performs the computations needed by the Adam method to minimize a (scalar-valued)
*loss function* \$J(\\theta)\$ by adjusting a set of *optimization parameters* \$\\theta\$.

The algorithm is described in [Adam](https://arxiv.org/pdf/1412.6980), using the comment just before
section 2.1 for a more efficient implementation.

# Parameters:
+ `graph::ComputationGraph`; Computation graph that is updated "in-place" by adding to it
        all the nodes needed to perform one step of gradient descent.
+ `loss::Node`: Scalar-valued computation node that corresponds to the loss function \$J(\\theta)\$
+ `theta::NamedTuple`: Named tuple with the variable nodes that correspond to the optimization
        parameters \$\\theta\\\$.

# Returns: named tuple the following nodes/tuples of nodes
+ `eta`: Scalar-valued variable node used to set the *learning rate* \$\\eta\$.
+ `beta1`: Scalar-valued variable node used to set Adam's beta1 parameter.
+ `beta2`: Scalar-valued variable node used to set Adam's beta2 parameter.
+ `epsilon`: Scalar-valued variable node used to set Adam's epsilon parameter.

+ `init_state`, `state`, `next_state`: Adam's internal state initializer, current value, and next
   value, which include the iteration number and the 2 moments

+ `next_theta::Tuple`: value \$\\theta^+\$ of the optimization parameters *after one gradient
        descent* iteration

+ `gradients`: gradients of the loss function with respect to the different variables in `theta`.

# Example:

"""
function adam!(
    graph::ComputationGraph;
    loss::Node,
    theta::NamedTuple,
    code::Union{Code,Nothing}=nothing,
) where {Node<:AbstractNode}
    # create solver parameters
    eta = variable(graph, fill(graph.TypeValue(1e-3)))
    beta1 = variable(graph, fill(graph.TypeValue(0.9)))
    beta2 = variable(graph, fill(graph.TypeValue(0.999)))
    epsilon = variable(graph, fill(graph.TypeValue(1e-8)))

    # create solver state variables
    iteration = variable(graph, ())
    moment1 = (; (
        k => variable(graph, size(v))
        for (k, v) in pairs(theta))...)
    moment2 = (; (
        k => variable(graph, size(v))
        for (k, v) in pairs(theta))...)

    # compute gradients
    gradients = (; (
        k => D(graph, loss, v)
        for (k, v) in pairs(theta))...)

    one_ = constant(graph, one(graph.TypeValue))

    # initialization for solver state variables
    init_iteration = one_
    init_moment1 = (; (
        k => zeros(graph, size(m))
        for (k, m) in pairs(moment1))...)
    init_moment2 = (; (
        k => zeros(graph, size(m))
        for (k, m) in pairs(moment2))...)

    # state updates
    next_iteration = @add graph iteration + one_

    next_moment1 = (; (
        k => @add graph beta1 * moment1[k] + (one_ - beta1) * grad
        for (k, grad) in pairs(gradients))...)

    next_moment2 = (; (
        k => @add graph beta2 * moment2[k] + (one_ - beta2) * (grad .* grad)
        for (k, grad) in pairs(gradients))...)

    η_iteration = @add graph eta * (
        sqrt(one_ - beta2^iteration) ./ (one_ - beta1^iteration))
    next_theta = (; (
        k => @add graph theta[k] - η_iteration *
                                   (next_moment1[k] ./ (epsilon .+ sqrt(next_moment2[k])))
        for k in eachindex(theta))...)
    #=
    hat_moment1 = [divideScalar(graph, next_moment1[k], -(graph, one_, ^(graph, adam_.beta1, t_))
    ) for k in eachindex(theta)]
    hat_moment2 = [divideScalar(graph, next_moment2[k], -(graph, one_, ^(graph, adam_.beta2, t_))
    ) for k in eachindex(theta)]
    next_theta = [-(graph,
        theta[k],
        scalarTimes(graph, adam_.η, pointDivide(graph,
            hat_moment1[k],
            scalarPlus(graph, adam_.epsilon, sqrt(graph, hat_moment2[k]))
        ))
    ) for k in eachindex(theta)]
    =#

    init_state = (; iteration=init_iteration,
        (Symbol(:moment1_, k) => m for (k, m) in pairs(init_moment1))...,
        (Symbol(:moment2_, k) => m for (k, m) in pairs(init_moment2))...,
    )
    state = (; iteration,
        (Symbol(:moment1_, k) => m for (k, m) in pairs(moment1))...,
        (Symbol(:moment2_, k) => m for (k, m) in pairs(moment2))...,
    )
    next_state = (; iteration=next_iteration,
        (Symbol(:moment1_, k) => m for (k, m) in pairs(next_moment1))...,
        (Symbol(:moment2_, k) => m for (k, m) in pairs(next_moment2))...,
    )

    if !isnothing(code)
        ## Add code
        sets!(code,
            eta => "setAdam_eta!",
            beta1 => "setAdam_beta1!",
            beta2 => "setAdam_beta2!",
            epsilon => "setAdam_epsilon!",
            iteration => "setIteration!")
        gets!(code,
            (; next_moments..., next_theta...) => "getNext")
        copies!(code,
            (zero_moments => moments) => "resetMoments!",
            ((; next_moments..., next_theta...) => (; moments..., theta...)) => "update!")
    end

    return (;
        eta, beta1, beta2, epsilon,
        init_state, state, next_state,
        next_theta, gradients,
    )
end
