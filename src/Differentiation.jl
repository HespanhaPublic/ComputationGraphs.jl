export D

"""
    Y = D(graph, F, P)
    Y = D(graph, V, F, P)

Computes the partial derivative of the expression encoded in the node `F` with respect to the
variable encoded in the node `P`, along the direction `V`. Formally, `Y` is a scalar/vector/matrix
with the same size as the variable `F`, with its `j`th entry equal to

``Y[j] = \\sum_i V[i] \\nabla_{P[j]} F[i]``

where ``\\nabla_{X[j]} F[i]`` the partial derivative of the `i`th entry of `F` with respect to the
`j`th entry of `P`.

The direction `V` can be omitted when `F` is a scalar, in which case 

``Y[j] = \\nabla_{P[j]} F``


# Parameters
+ `graph::ComputationGraph`: Computation graph encoding the relevant expressions and variables.
+ `V::Node`: Direction with respect the partial derivative is computed. 
        This node needs to have the same size as `F`.
+ `F::Node`: Expression to be differentiated.
+ `P::NodeVariable`: Variable with respect to `F` will be differentiated. 
        This node must have been created using [variable](@ref)

# Returns 
+ `Y::Node`: Node that encodes the expression of the partial derivative (added to the graph if it
        was not already part of it.) 
        This node will have the same size as `P`.
"""
ComputationGraphs.D

# d Variable 
D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeVariable,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode} =
    (F === P) ? V : zeros(graph, size(P))

# d Constant 
D(graph::ComputationGraph{TypeValue},
    ::NodeV,
    ::NodeC,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode,NodeC<:AbstractConstantNode} = zeros(graph, size(P))

#########
## losses
#########

# d norm2 
function D(graph::ComputationGraph{TypeValue},
    F::NodeNorm2,
    P::NodeVariable,
) where {TypeValue}
    Y = graph.nodes[F.parentIds[1]]
    two = constant(graph, convert(eltype(Y), 2))
    if false
        D1 = D(graph, Y, Y, P)
        Dout = scalarTimes(graph, two, D1)
    else
        # faster
        Y2 = scalarTimes(graph, two, Y)
        Dout = D(graph, Y2, Y, P)
    end
    @assert size(Dout) == size(P) "mismatch in D(norm2) $(size(Dout)) != $(size(P))"
    return Dout
end
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeNorm2,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    Y = graph.nodes[F.parentIds[1]]
    two = pointTimes(graph, constant(graph, TypeValue(2)), V)
    if false
        D1 = D(graph, Y, Y, P)
        Dout = scalarTimes(graph, two, D1)
    else
        # faster
        Y2 = scalarTimes(graph, two, Y)
        Dout = D(graph, Y2, Y, P)
    end
    @assert size(Dout) == size(P) "mismatch in D(norm2) $(size(Dout)) != $(size(P))"
    return Dout
end

# d norm1
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeNorm1,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    Y = graph.nodes[F.parentIds[1]]
    barV = scalarTimes(graph, V, sign(graph, Y))
    Dout = D(graph, barV, Y, P)
    @assert size(Dout) == size(P) "mismatch in D(norm1) $(size(Dout)) != $(size(P))"
    return Dout
end
function D(graph::ComputationGraph{TypeValue},
    F::NodeNorm1,
    P::NodeVariable,
) where {TypeValue}
    Y = graph.nodes[F.parentIds[1]]
    barV = sign(graph, Y)
    Dout = D(graph, barV, Y, P)
    @assert size(Dout) == size(P) "mismatch in D(norm1) $(size(Dout)) != $(size(P))"
    return Dout
end

# d huber
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeHuber,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    Y = graph.nodes[F.parentIds[1]]
    barV = scalarTimes(graph, V, sat(graph, Y))
    Dout = D(graph, barV, Y, P)
    @assert size(Dout) == size(P) "mismatch in D(hubber) $(size(Dout)) != $(size(P))"
    return Dout
end
function D(graph::ComputationGraph{TypeValue},
    F::NodeHuber,
    P::NodeVariable,
) where {TypeValue}
    Y = graph.nodes[F.parentIds[1]]
    barV = sat(graph, Y)
    Dout = D(graph, barV, Y, P)
    @assert size(Dout) == size(P) "mismatch in D(hubber) $(size(Dout)) != $(size(P))"
    return Dout
end

############
## Reshaping
############

# d adjoint (x')
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeAdjoint_,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    @assert size(V) == size(F) "mismatch between V[$(typeof(V)),$(size(V))] and F[$(typeof(F)),$(size(F))]"
    x = graph.nodes[F.parentIds[1]]
    VT = adjoint(graph, V)
    Dout = D(graph, VT, x, P)
    @assert size(Dout) == size(P) "mismatch in D(adjoint) $(size(Dout)) != $(size(P))"
    return Dout
end

# d selectRows(A,rows)
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeSelectRows,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    @assert size(V) == size(F) "mismatch between V[$(typeof(V)),$(size(V))] and F[$(typeof(F)),$(size(F))]"

    A = graph.nodes[F.parentIds[1]]
    rows = graph.nodes[F.parentIds[2]]

    Vm = expandColumns(graph, V, rows, size(A, 1))
    Dout = D(graph, Vm, A, P) # TODO probably wasteful to compute derivative for the whole A
    @assert size(Dout) == size(P) "mismatch in D(selectRows) $(size(Dout)) != $(size(P))"
    return Dout
end


#######################
## Addition/subtraction
#######################

# d add (a+b)
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodePlus,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    @assert size(V) == size(F) "mismatch between V[$(typeof(V)),$(size(V))] and F[$(typeof(F)),$(size(F))]"

    a = graph.nodes[F.parentIds[1]]
    b = graph.nodes[F.parentIds[2]]
    D1 = D(graph, V, a, P)
    D2 = D(graph, V, b, P)
    Dout = +(graph, D1, D2)
    @assert size(Dout) == size(P) "mismatch in D(plus) $(size(Dout)) != $(size(P))"
    return Dout
end


# d subtract (a-b)
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeSubtract,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    @assert size(V) == size(F) "mismatch between V[$(typeof(V)),$(size(V))] and F[$(typeof(F)),$(size(F))]"

    a = graph.nodes[F.parentIds[1]]
    b = graph.nodes[F.parentIds[2]]
    D1 = D(graph, V, a, P)
    D2 = D(graph, V, b, P)
    Dout = -(graph, D1, D2)
    @assert size(Dout) == size(P) "mismatch in D(subtract) $(size(Dout)) != $(size(P))"
    return Dout
end

###########
## Products
###########

# d pointTimes (a .* b)
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodePointTimes,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    @assert size(V) == size(F) "mismatch between V[$(typeof(V)),$(size(V))] and F[$(typeof(F)),$(size(F))]"
    a = graph.nodes[F.parentIds[1]]
    b = graph.nodes[F.parentIds[2]]
    Va = pointTimes(graph, V, a)
    Vb = pointTimes(graph, V, b)
    D1 = D(graph, Va, b, P)
    D2 = D(graph, Vb, a, P)
    Dout = +(graph, D1, D2)
    @assert size(Dout) == size(P) "mismatch in D(pointTimes) $(size(Dout)) != $(size(P))"
    return Dout
end

# d scalarTimes (a .* b, a is scalar)
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeScalarTimes,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    @assert size(V) == size(F) "mismatch between V[$(typeof(V)),$(size(V))] and F[$(typeof(F)),$(size(F))]"
    a = graph.nodes[F.parentIds[1]]
    b = graph.nodes[F.parentIds[2]]
    Va = scalarTimes(graph, a, V)
    Vb = dot(graph, V, b)
    D1 = D(graph, Va, b, P)
    D2 = D(graph, Vb, a, P)
    Dout = +(graph, D1, D2)
    @assert size(Dout) == size(P) "mismatch in D(scalarTimes) $(size(Dout)) != $(size(P))"
    return Dout
end

# d timesAdjoint (x * y')
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeTimesAdjoint,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    @assert size(V) == size(F) "mismatch between V[$(typeof(V)),$(size(V))] and F[$(typeof(F)),$(size(F))]"
    x = graph.nodes[F.parentIds[1]]
    y = graph.nodes[F.parentIds[2]]
    xV = adjointTimes(graph, x, V) # x' * V
    Vy = times(graph, V, y)        # V * y
    yT = adjoint(graph, y)
    D1 = D(graph, xV, yT, P)
    D2 = D(graph, Vy, x, P)
    Dout = +(graph, D1, D2)
    @assert size(Dout) == size(P) "mismatch in D(timesAdjoint) $(size(Dout)) != $(size(P))"
    return Dout
end

# d timesAdjointOnes (x * ones')
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeTimesAdjointOnes,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    @assert size(V) == size(F) "mismatch between V[$(typeof(V)),$(size(V))] and F[$(typeof(F)),$(size(F))]"
    x = graph.nodes[F.parentIds[1]]
    o = ones(graph, F.parameters[1]) # ones
    Vy = times(graph, V, o)        # V * y
    Dout = D(graph, Vy, x, P)
    @assert size(Dout) == size(P) "mismatch in D(timesAdjointOnes) size(Dout)=$(size(Dout)) != size(P)=$(size(P))"
    return Dout
end

# d adjointTimes (A' * x)
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeAdjointTimes,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    @assert size(V) == size(F) "mismatch between V[$(typeof(V)),$(size(V))] and F[$(typeof(F)),$(size(F))]"
    A = graph.nodes[F.parentIds[1]]
    x = graph.nodes[F.parentIds[2]]
    Av = times(graph, A, V)        # A * V
    Vx = timesAdjoint(graph, V, x) # V * x'
    AT = adjoint(graph, A)
    D1 = D(graph, Av, x, P)
    D2 = D(graph, Vx, AT, P)
    Dout = +(graph, D1, D2)
    @assert size(Dout) == size(P) "mismatch in D(adjointTimes) $(size(Dout)) != $(size(P))"
    return Dout
end

# d times (A * x)
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeTimes,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    @assert size(V) == size(F) "mismatch between V[$(typeof(V)),$(size(V))] and F[$(typeof(F)),$(size(F))]"

    A = graph.nodes[F.parentIds[1]]
    x = graph.nodes[F.parentIds[2]]

    Av = adjointTimes(graph, A, V) # A' * V
    Vx = timesAdjoint(graph, V, x) # V * x'
    D1 = D(graph, Av, x, P)
    D2 = D(graph, Vx, A, P)
    Dout = +(graph, D1, D2)
    @assert size(Dout) == size(P) "mismatch in D(Times) $(size(Dout)) != $(size(P))"
    return Dout
end

# d affine (A*x+b)
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeAffine,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    @assert size(V) == size(F) "mismatch between V[$(typeof(V)),$(size(V))] and F[$(typeof(F)),$(size(F))]"

    A = graph.nodes[F.parentIds[1]]
    x = graph.nodes[F.parentIds[2]]
    b = graph.nodes[F.parentIds[3]]

    Av = adjointTimes(graph, A, V) # A' * V
    Vx = timesAdjoint(graph, V, x) # V * x'
    D1 = D(graph, Av, x, P)
    D2 = D(graph, Vx, A, P)
    if size(V, 2) > 1
        D3 = D(graph, sumColumns(graph, V), b, P)
    else
        D3 = D(graph, V, b, P)
    end
    D12 = +(graph, D1, D2)
    Dout = +(graph, D12, D3)
    @assert size(Dout) == size(P) "mismatch in D(affine) $(size(Dout)) != $(size(P))"
    return Dout
end

#=
# d affineRows (A*x+b)
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeAffineRows{Tuple{},Tuple{Int,Int,Int},Tuple{TP1V,TP2V,TP3V},VF},
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode,
    TP1V<:AbstractArray,TP2V<:AbstractArray,TP3V<:AbstractArray}
    @assert size(V) == size(F) "mismatch between V[$(typeof(V)),$(size(V))] and F[$(typeof(F)),$(size(F))]"

    A = graph.nodes[F.parentIds[1]]
    x = graph.nodes[F.parentIds[2]]
    b = graph.nodes[F.parentIds[3]]
    rows = F.parents[4]

    brows = selectRows(graph, b, rows)
    Av = rowsAdjointTimes(graph, A, V, rows) # A[rows,:]' * V
    Vx = timesAdjoint(graph, V, x) # V * x'
    D1 = D(graph, Av, x, P)
    D2 = D(graph, Vx, A, P)
    D3 = D(graph, V, brows, P)
    D12 = +(graph, D1, D2)
    Dout = +(graph, D12, D3)
    @assert size(Dout) == size(P) "mismatch in D(affineRows) $(size(Dout)) != $(size(P))"
    return Dout
end
=#

############
## Divisions
############

# d a scalar / b scalar = (b a'-a b')/b^2
function D(graph::ComputationGraph{TypeValue},
    F::NodeDivideScalar,
    P::NodeVariable,
) where {TypeValue}
    a = graph.nodes[F.parentIds[1]]
    b = graph.nodes[F.parentIds[2]]
    @assert size(a) == ()
    @assert size(b) == ()
    two = constant(graph, convert(eltype(b), 2))
    bDa = D(graph, b, a, P)
    aDb = D(graph, a, b, P)
    Dout = divideScalar(graph, -(graph, bDa, aDb), ^(graph, b, two))
    @assert size(Dout) == size(P) "mismatch in D(scalar/scalar) $(size(Dout)) != $(size(P))"
    return Dout
end

##################
## Maximum/Minimum
##################

# d maxRow(A) 
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeMaxRow,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    A = graph.nodes[F.parentIds[1]]

    rows = findMaxRow(graph, A)
    Vm = expandColumns(graph, V, rows, size(A, 1)) # TODO wasteful to expand with zeros
    Dout = D(graph, Vm, A, P)

    @assert size(Dout) == size(P) "mismatch in D(maxRow/scalar) $(size(Dout)) != $(size(P))"
    return Dout
end


############
## Functions
############

# d relu (x)
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeRelu,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    @assert size(V) == size(F) "mismatch between V[$(typeof(V)),$(size(V))] and F[$(typeof(F)),$(size(F))]"
    x = graph.nodes[F.parentIds[1]]
    H = heaviside(graph, x)
    VH = pointTimes(graph, V, H) # V .* H
    Dout = D(graph, VH, x, P)
    @assert size(Dout) == size(P) "mismatch in D(relu) $(size(Dout)) != $(size(P))"
    return Dout
end

# d exp (x)
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeExp_,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    @assert size(V) == size(F) "mismatch between V[$(typeof(V)),$(size(V))] and F[$(typeof(F)),$(size(F))]"
    x = graph.nodes[F.parentIds[1]]
    H = exp(graph, x)
    VH = pointTimes(graph, V, H) # V .* H
    Dout = D(graph, VH, x, P)
    @assert size(Dout) == size(P) "mismatch in D(exp) $(size(Dout)) != $(size(P))"
    return Dout
end

# d logistic: f(x)=1/(1+exp(-x))   f'(x)=exp(-x)/(1+exp(-x))^2
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeLogistic_,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    @assert size(V) == size(F) "mismatch between V[$(typeof(V)),$(size(V))] and F[$(typeof(F)),$(size(F))]"
    x = graph.nodes[F.parentIds[1]]
    H = dlogistic(graph, x)
    VH = pointTimes(graph, V, H) # V .* H
    Dout = D(graph, VH, x, P)
    @assert size(Dout) == size(P) "mismatch in D(logistic) $(size(Dout)) != $(size(P))"
    return Dout
end

# d^2 logistic: f(x)=1/(1+exp(-x))   f''(x)=(exp(-2x)-exp(-x)) /(1+exp(-x))^3
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeDlogistic,
    P::NodeVariable,
) where {TypeValue,NodeV<:AbstractNode}
    @assert size(V) == size(F) "mismatch between V[$(typeof(V)),$(size(V))] and F[$(typeof(F)),$(size(F))]"
    x = graph.nodes[F.parentIds[1]]
    H = ddlogistic(graph, x)
    VH = pointTimes(graph, V, H) # V .* H
    Dout = D(graph, VH, x, P)
    @assert size(Dout) == size(P) "mismatch in D2(logistic) $(size(Dout)) != $(size(P))"
    return Dout
end


##########
## Hessian
##########

export hessian

"""
    Y = hessian(graph, F, P, Q)

Computes the Hessian matrix of the expression encoded in the (scalar-valued) node `F` with respect
to the variables encoded in the (vector-values) nodes `P` and `Q`. Formally, `Y` is a matrix with its `(i,j)`th entry equal to

``Y[i,j] = \\nabla_{P[i]} \\nabla_{Q[j]} F``

where ``\\nabla_{X}`` denotes partial derivative with respect to `X`.

# Parameters
+ `graph::ComputationGraph`: Computation graph encoding the relevant expressions and variables.
+ `F::Node`: Expression to be differentiated.
+ `P::NodeVariable`: First variable with respect to `F` will be differentiated. 
        This node must have been created using [variable](@ref)
+ `Q::NodeVariable`: Second variable with respect to `F` will be differentiated. 
        This node must have been created using [variable](@ref)

# Returns 
+ `Y::Node`: Node that encodes the expression of the Hessian matrix (added to the graph if it
        was not already part of it.) 
"""
function hessian(graph::ComputationGraph{TypeValue},
    F::NodeF,
    P::NodeVariable,
    Q::NodeVariable,
) where {TypeValue,NodeF<:AbstractNode}
    @printf("Hessian of [%s] with respect to [[%s], [%s]]\n", size(F), size(P), size(Q))

    @assert size(F) == () "Hessian only available for scalars"

    @assert length(eachindex(nodeValue(Q))) <= length(eachindex(nodeValue(P))) "Hessian: reverse order of variables for efficiency"
    #fP = D(graph, ones(graph, size(F)), F, P)
    fQ = D(graph, F, Q)
    Dout = [D(graph, unitvector(graph, size(Q), k), fQ, P)
            for k in eachindex(nodeValue(Q))]
    return Dout
end

## Catch all error
function D(graph::ComputationGraph{TypeValue},
    V::NodeV,
    F::NodeF,
    P::NodeP,
) where {TypeValue,NodeV<:AbstractNode,NodeF<:AbstractNode,NodeP<:AbstractNode}
    @show @red(string(typeof(F)))
    @printf("  V = %s\n  F = \e[31m%s\e[39m\n  P = %s\n",
        typeof(V), typeof(F), typeof(P)) # not using @red because of {}
    throw(DomainError(typeof(F), "missing derivative"))
end

function D(graph::ComputationGraph{TypeValue},
    F::NodeF,
    P::NodeP,
) where {TypeValue,NodeF<:AbstractNode,NodeP<:AbstractNode}
    @show @red(string(typeof(F)))
    @printf("  F = \e[31m%s\e[39m\n  P = %s\n",
        typeof(F), typeof(P)) # not using @red because of {}
    throw(DomainError(typeof(F), "missing derivative"))
end
