using MacroTools # debug

import LinearAlgebra

# https://juliasimd.github.io/LoopVectorization.jl/stable/examples/matrix_vector_ops/
import LoopVectorization

# https://octavian.julialinearalgebra.org/stable/
import Octavian

if true
    # parallelism: good for computeSpawn! and okay (but not best) for Polyester
    # trainNNqlearning_Dubins: computeSpawn! good (25ms)
    const MATRIX_MULT = :BLAS
    @inline mul!(C, A, B) = LinearAlgebra.mul!(C, A, B)
    @inline mul!(C, A, B, α, β) = LinearAlgebra.mul!(C, A, B, α, β)
elseif true
    # parallelism: best for computeSpawn! and Polyester
    # trainNNqlearning_Dubins: computeSpawn! inconsistent??? (sometimes 35 others 24ms)
    const MATRIX_MULT = :OCTAVIAN_1THREAD
    @inline mul!(C, A, B) =
        Octavian.matmul!(C, A, B, one(eltype(A)), zero(eltype(C)), 1) # single thread
    @inline mul!(C, A, B, α, β) = Octavian.matmul!(C, A, B, α, β, 1) # single thread
elseif false
    # bad for computeSpawn!, but good for Polyester
    const MATRIX_MULT = :OCTAVIAN_AUTOTHREADS
    @inline mul!(C, A, B) = Octavian.matmul!(C, A, B)
    @inline mul!(C, A, B, α, β) = Octavian.matmul!(C, A, B, α, β)
end

##################
## Unary operators
##################

@doc "norm2() computes the sum of the square values of a vector or matrix" norm2
@newnode norm2{x}::()
#@macroexpand @newnode norm2{x}::()
@inline function cg_norm2!(
    val::TypeArray{TypeValue,0}, x::TypeArray{TypeValue,N}
) where {TypeValue,N}
    val .= sum(abs2, x)
    return nothing
end

@doc "norm1() computes the sum of the absolute values of a vector or matrix" norm1
@newnode norm1{x}::()
@inline function cg_norm1!(
    val::TypeArray{TypeValue,0}, x::TypeArray{TypeValue,N}
) where {TypeValue,N}
    val .= sum(abs, x)
    return nothing
end

@doc "huber() computes the huber loss of a vector or matrix" huber
@newnode huber{x}::()
@inline function cg_huber!(
    val::TypeArray{TypeValue,0}, x::TypeArray{TypeValue,N}
) where {TypeValue,N}
    val[1] = zero(TypeValue)
    for i in eachindex(x)
        @inbounds val[1] += (abs(x[i]) <= one(TypeValue)) ?
                            abs2(x[i]) * TypeValue(0.5) :
                            abs(x[i]) - TypeValue(0.5)
    end
    return nothing
end

@doc "minus() unitary minus of a vector or matrix" minus
@newnode minus{x}::(size(x))
@inline function cg_minus!(
    val::TypeArray{TypeValue,N}, x::TypeArray{TypeValue,N}
) where {TypeValue,N}
    #@. val = -x
    @simd for k in eachindex(val, x)
        @inbounds val[k] = -x[k]
    end
    return nothing
end
# overload usual Base.-
"""-() unitary minus operator for a vector or matrix"""
Base.:-(graph::ComputationGraph{TypeValue}, a::Node
) where {TypeValue,Node<:AbstractNode} = minus(graph, a)


@doc "adjoint() computes adjoint/transpose of a vector or matrix" adjoint_
@newnode adjoint_{x}::(length(size(x)) == 1 ? (1, size(x, 1)) : (size(x, 2), size(x, 1)))
@inline function cg_adjoint_!(
    val::TypeArray{TypeValue,2}, x::TypeArray{TypeValue,N}
) where {TypeValue,N}
    adjoint!(val, x)
    return nothing
end
# overload usual Base.adjoint
"""
adjoint() computes adjoint/transpose of a vector or matrix
"""
Base.adjoint(graph::ComputationGraph{TypeValue}, a::Node
) where {TypeValue,Node<:AbstractNode} = adjoint_(graph, a)

@doc "column(A,k) returns the column k of A as a vector" column
@newnode column{1,A}::(size(A, 1),)
@inline function cg_column!(
    val::TypeArray{TypeValue,1}, A::TypeArray{TypeValue,2}, k::Int
) where {TypeValue}
    copyto!(val, @view A[:, k])
    return nothing
end

# FIXME: if A is a vector, then it should not create a new node
@doc "sumColumns(A) returns a vector with the sums of the columns of a matrix A" sumColumns
@newnode sumColumns{A}::(size(A, 1),)
@inline function cg_sumColumns!(
    val::TypeArray{TypeValue,1}, A::TypeArray{TypeValue,2}
) where {TypeValue}
    @assert length(val) == size(A, 1)
    copyto!(val, @view A[:, 1])
    for j in 2:size(A, 2)
        @simd for i in axes(A, 1)
            @inbounds val[i] += A[i, j]
        end
    end
    return nothing
end
sumColumns(A::AbstractMatrix) = sum(A; dims=2)

@doc "sumColumns(ExpandColumns(x,rows,nRows))" sumExpandColumns
@newnode sumExpandColumns{1,x,rows}::(par1,)
@inline function cg_sumExpandColumns!(
    val::TypeArray{TypeValue,1},
    x::TypeArray{TypeValue,1},
    rows::TypeArray{Int,1},
    ::Int # nRows not needed since encoded in size
) where {TypeValue}
    fill!(val, zero(TypeValue))
    for (i, r) in enumerate(rows)
        val[r] += x[i]
    end
    return nothing
end

#######################
## Addition/Subtraction
#######################

@doc "a + b addition operator" plus
@newnode plus{a,b}::size(a)
@inline function cg_plus!(
    val::TypeArray{TypeValue,N}, a::TypeArray{TypeValue,N}, b::TypeArray{TypeValue,N}
) where {TypeValue,N}
    #@. val = a+b
    @simd for k in eachindex(val, a, b)
        @inbounds val[k] = a[k] + b[k]
    end
    return nothing
end
# overload usual Base.:+
"""a + b addition operator"""
Base.:+(graph::ComputationGraph{TypeValue}, a::Node1, b::Node2
) where {TypeValue,Node1<:AbstractNode,Node2<:AbstractNode} = plus(graph, a, b)


@doc "a - b subtraction operator" subtract
@newnode subtract{a,b}::size(a)
@inline function cg_subtract!(
    val::TypeArray{TypeValue,N}, a::TypeArray{TypeValue,N}, b::TypeArray{TypeValue,N}
) where {TypeValue,N}
    #@. val = a-b

    ## BLAS option does not seem faster
    copyto!(val, a)
    LinearAlgebra.BLAS.axpy!(-one(TypeValue), b, val) # N=1

    ## loop option
    #@simd for k in eachindex(val, a, b)
    #    @inbounds val[k] = a[k] - b[k]
    #end
    return nothing
end
# overload usual Base.:
"""a - b subtraction operator"""
Base.:-(graph::ComputationGraph{TypeValue}, a::Node1, b::Node2
) where {TypeValue,Node1<:AbstractNode,Node2<:AbstractNode} = subtract(graph, a, b)

@doc "scalarPlus(a, b) = a .+ b, where a is a scalar" scalarPlus
@newnode scalarPlus{a,b}::size(b)
@inline function cg_scalarPlus!(
    val::TypeArray{TypeValue,N}, a::TypeArray{TypeValue,0}, b::TypeArray{TypeValue,N}
) where {TypeValue,N}
    #@. val = a + b

    ## loop option
    @simd for k in eachindex(val, b)
        @inbounds val[k] = a[1] + b[k]
    end
    return nothing
end

@doc "columnPlus(a, b) = a .+ b, where a is a vector with the same number of rows as the matrix b" columnPlus
@newnode columnPlus{a,b}::size(b)
@inline function cg_columnPlus!(
    val::TypeArray{TypeValue,N}, a::TypeArray{TypeValue,1}, b::TypeArray{TypeValue,N}
) where {TypeValue,N}
    #@. val = a + b

    ## loop option
    @assert size(val) == size(b)
    @assert size(val, 1) == length(a)
    @simd for i in axes(val, 1)
        @simd for j in axes(val, 2)
            @inbounds val[i, j] = a[i] + b[i, j]
        end
    end
    return nothing
end

@doc "rowPlus(a, b) = a .+ b, where a is a vector with the same number of columns as the matrix b" rowPlus
@newnode rowPlus{a,b}::size(b)
@inline function cg_rowPlus!(
    val::TypeArray{TypeValue,N}, a::TypeArray{TypeValue,1}, b::TypeArray{TypeValue,N}
) where {TypeValue,N}
    #@. val = a + b

    ## loop option
    @assert size(val) == size(b)
    @assert size(val, 2) == length(a)
    @simd for i in axes(val, 1)
        @simd for j in axes(val, 2)
            @inbounds val[i, j] = a[j] + b[i, j]
        end
    end
    return nothing
end

# overload broadcast of +
"""
a .+ b broadcast, which maps to plus() or scalarPlus() depending on the sizes of the arguments
"""
function Broadcast.broadcasted(
    ::typeof(+),
    graph::ComputationGraph{TypeValue},
    a::Node1,
    b::Node2
) where {TypeValue,Node1<:AbstractNode,Node2<:AbstractNode}
    if size(a) == size(b)
        # regular sum
        return plus(graph, a, b)
    elseif size(a) == ()
        # add to scalar
        return scalarPlus(graph, a, b)
    elseif size(b) == ()
        # add to scalar
        return scalarPlus(graph, b, a)
    elseif length(size(a)) == 1 && length(size(b)) == 2 &&
           size(a, 1) == size(b, 1) && size(b, 1) != size(b, 2)
        # add to vector
        return columnPlus(graph, a, b)
    elseif length(size(a)) == 1 && length(size(b)) == 2 &&
           size(a, 1) == size(b, 2) && size(b, 1) != size(b, 2)
        # add to vector
        return rowPlus(graph, a, b)
    elseif length(size(a)) == 1 && length(size(b)) == 2 &&
           size(a, 1) == size(b, 1) && size(b, 1) == size(b, 2)
        error("ambiguous broadcasting vector + square matrix")
    elseif length(size(b)) == 1 && length(size(a)) == 2 &&
           size(b, 1) == size(a, 1) && size(a, 1) != size(a, 2)
        # add to vector
        return columnPlus(graph, b, a)
    elseif length(size(b)) == 1 && length(size(a)) == 2 &&
           size(b, 1) == size(a, 2) && size(a, 1) != size(a, 2)
        # add to vector
        return rowPlus(graph, b, a)
    elseif length(size(b)) == 1 && length(size(a)) == 2 &&
           size(b, 1) == size(a, 1) && size(a, 1) == size(a, 2)
        error("ambiguous broadcasting vector + square matrix")
    else
        display(a)
        display(b)
        error(".+ not implements for nodes of these sizes")
    end
end

# overload broadcast of -
"""
a .- b broadcast maps to subtract()
"""
function Broadcast.broadcasted(
    ::typeof(-),
    graph::ComputationGraph{TypeValue},
    a::Node1,
    b::Node2
) where {TypeValue,Node1<:AbstractNode,Node2<:AbstractNode}
    if size(a) == size(b)
        # regular subtract
        return subtract(graph, a, b)
    else
        display(a)
        display(b)
        error(".- not implements for nodes of these sizes")
    end
end


###########
## Products
###########

@doc "scalarTimes(a,M)= a .* M computes the product of a scalar a by a matrix M" scalarTimes
@newnode scalarTimes{a,M}::size(M)
@inline function cg_scalarTimes!(
    val::TypeArray{TypeValue,N}, a::TypeArray{TypeValue,0}, M::TypeArray{TypeValue,N}
) where {TypeValue,N}
    #@. val = a * M

    ## BLAS option does not seem faster
    #copyto!(val, M)
    #LinearAlgebra.BLAS.scal!(a[1], val)

    ## FIXME loop option does not use threads
    @simd for k in eachindex(val, M)
        @inbounds val[k] = a[1] * M[k]
    end
    return nothing
end

@doc "pointTimes(a, b) = a .* b" pointTimes
@newnode pointTimes{a,b}::size(a)
@inline function cg_pointTimes!(
    val::TypeArray{TypeValue,N}, a::TypeArray{TypeValue,N}, b::TypeArray{TypeValue,N}
) where {TypeValue,N}
    #@. val = a * b
    ## FIXME loop option does not use threads
    @simd for k in eachindex(val, a, b)
        @inbounds val[k] = a[k] * b[k]
    end
    return nothing
end

# overload broadcast of *
"""
a .* b broadcast maps to pointTimes() or scalarTimes() depending on the sizes of the arguments
"""
function Broadcast.broadcasted(
    ::typeof(*),
    graph::ComputationGraph{TypeValue},
    a::Node1,
    b::Node2
) where {TypeValue,Node1<:AbstractNode,Node2<:AbstractNode}
    if size(a) == size(b)
        return pointTimes(graph, a, b)
    elseif size(a) == ()
        return scalarTimes(graph, a, b)
    elseif size(b) == ()
        return scalarTimes(graph, b, a)
    else
        display(a)
        display(b)
        error(".* not implements for nodes of these sizes")
    end
end

@doc "times(A,x) computes the product of a matrix A by a matrix/vector x" times
## *(A,x) = A*x
@newnode times{A,x}::(length(size(x)) == 1 ? (size(A, 1),) : (size(A, 1), size(x, 2)))

@inline function cg_times!(
    val::TypeArray{TypeValue,N}, A::TypeArray{TypeValue,2}, x::TypeArray{TypeValue,N}
) where {TypeValue,N}

    ## faster than loop
    mul!(val, A, x)

    #LinearAlgebra.BLAS.gemv!('N', one(TypeValue), A, x, zero(TypeValue), val) # restricted to N==1

    ## loop option is slower
    #fill!(val, zero(TypeValue))
    #for j in axes(A, 1)
    #    @simd for i in eachindex(x)
    #        @inbounds val[j] += A[j, i] * x[i]
    #    end
    #end
    return nothing
end

# overload regular *
"""a * b maps to times() or scalarTimes() depending on the sizes of the arguments"""
function Base.:*(
    graph::ComputationGraph{TypeValue},
    a::Node1,
    b::Node2
) where {TypeValue,Node1<:AbstractNode,Node2<:AbstractNode}
    if length(size(a)) == 2 && size(a, 2) == size(b, 1)
        return times(graph, a, b)
    elseif size(a) == ()
        return scalarTimes(graph, a, b)
    elseif size(b) == ()
        return scalarTimes(graph, b, a)
    else
        display(a)
        display(b)
        error("* not implements for these nodes of these sizes")
    end
end

@doc "adjointTimes(A,x)= A'*x computes the product of the adjoint of the matrix A with a matrix/vector x" adjointTimes
@newnode adjointTimes{A,x}::(length(size(x)) == 1 ? (size(A, 2),) : (size(A, 2), size(x, 2)))
@inline function cg_adjointTimes!(
    val::TypeArray{TypeValue,N},
    A::TypeArray{TypeValue,M}, # M could be 1 or 2
    x::TypeArray{TypeValue,N}
) where {TypeValue,N,M}
    ## faster than loop
    mul!(val, A', x)
    #LinearAlgebra.BLAS.gemv!('T', one(TypeValue), A, x, zero(TypeValue), val) # restricted to N==1

    ## loop option is slower
    #fill!(val, zero(TypeValue))
    #for j in axes(A, 2)
    #    @simd for i in eachindex(x)
    #        @inbounds val[j] += A[i, j] * x[i]
    #    end
    #end
    return nothing
end

@doc "adjointTimesExpandColumns(A,x,rows) = A'*expandColumns(x,rows) computes the product of the adjoint of the matrix A with expandColumns(x,rows)" adjointTimesExpandColumns
@newnode adjointTimesExpandColumns{A,x,rows}::(size(A, 2), length(x))
@inline function cg_adjointTimesExpandColumns!(
    val::TypeArray{TypeValue,2},
    A::TypeArray{TypeValue,2},
    x::TypeArray{TypeValue,1},
    rows::TypeArray{Int,1},
) where {TypeValue}
    for (i, r) in enumerate(rows)
        val[:, i] .= (x[i] .* @view A[r, :])
    end
    return nothing
end


# TODO seems like a good idea to be "lazy" about this, but in NN only appears in last operation of derivative w.r.t W
@doc "timesAdjoint(x, y') = x * y'" timesAdjoint
@newnode timesAdjoint{x,y}::(size(x, 1), size(y, 1))
@inline function cg_timesAdjoint!(
    val::TypeArray{TypeValue,2}, x::TypeArray{TypeValue,N}, y::TypeArray{TypeValue,N}
) where {TypeValue,N}
    ## faster than loop
    mul!(val, x, y')

    ## loop option
    #@assert size(val) == (length(x), length(y))
    #for (i, xi) in enumerate(x)
    #    @simd for j in eachindex(y)
    #        @inbounds val[i, j] = xi * y[j]  # FIXME unsafe
    #    end
    #end
    return nothing
end

@doc "unitTimesAdjoint(y,dims,k) = unitvector(dims,k)*y'" unitTimesAdjoint
@newnode unitTimesAdjoint{2,y}::(par1, size(y, 1))
@inline function cg_unitTimesAdjoint!(
    val::TypeArray{TypeValue,2},
    y::TypeArray{TypeValue,1},
    ::Int, # not needed since already in size of val
    k::Int,
) where {TypeValue}
    fill!((@view val[1:k-1, :]), zero(TypeValue))
    copyto!((@view val[k, :]), y)
    fill!((@view val[k+1:end, :]), zero(TypeValue))
    return nothing
end

@doc "expandColumnsTimesAdjoint(x,y,rows,nRows)=expandColumns(x,rows,nRows)*y'" expandColumnsTimesAdjoint
@newnode expandColumnsTimesAdjoint{1,x,y,rows}::(par1, size(y, 1))
@inline function cg_expandColumnsTimesAdjoint!(
    val::TypeArray{TypeValue,2},
    x::TypeArray{TypeValue,1},
    y::TypeArray{TypeValue,2},
    rows::TypeArray{Int,1},
    ::Int, # nRows not needed since encoded into size
) where {TypeValue}
    fill!(val, zero(TypeValue))
    # val = A*y = sum_i A[:,i]*y[i,:] but A[:,i] only has r-entry nonzero & need adjoint of y
    for (i, r) in enumerate(rows)
        for j in axes(val, 2)
            val[r, j] += x[i] * y[j, i]
        end
        #val[r, :] .+= x[i] .* @view y[:, i]
    end
    return nothing
end

@doc "timesAdjointOnes(x,n)=x*ones(n)" timesAdjointOnes
@newnode timesAdjointOnes{1,x}::(size(x, 1), par1)
@inline function cg_timesAdjointOnes!(
    val::TypeArray{TypeValue,2},
    x::TypeArray{TypeValue,1},
    ::Int   # not needed since already in size of val
) where {TypeValue}
    @simd for k in axes(val, 2)
        @inbounds copyto!((@view val[:, k]), x)
    end
    return nothing
end

@doc "dot(x,y) computes the inner product of two vectors" dot
@newnode dot_{a,b}::()
@inline function cg_dot_!(
    val::TypeArray{TypeValue,0}, a::TypeArray{TypeValue,N}, b::TypeArray{TypeValue,N}
) where {TypeValue,N}
    #val = dot(a,b)
    val[1] = dot(a, b)
    return nothing
end
# overload usual LinearAlgebra.:
LinearAlgebra.dot(graph::ComputationGraph{TypeValue}, a::Node1, b::Node2
) where {TypeValue,Node1<:AbstractNode,Node2<:AbstractNode} = dot_(graph, a, b)

############
## Divisions
############

@doc "pointDivide(a, b) = a ./ b" pointDivide
@newnode pointDivide{a,b}::size(a)
@inline function cg_pointDivide!(
    val::TypeArray{TypeValue,N}, a::TypeArray{TypeValue,N}, b::TypeArray{TypeValue,N}
) where {TypeValue,N}
    #@. val = a * b
    @simd for k in eachindex(val, a, b)
        @inbounds val[k] = a[k] / b[k]
    end
    return nothing
end

@doc "divideScalar(a, b) = a ./ b, where b is a scalar" divideScalar
@newnode divideScalar{a,b}::size(a)
@inline function cg_divideScalar!(
    val::TypeArray{TypeValue,N}, a::TypeArray{TypeValue,N}, b::TypeArray{TypeValue,0}
) where {TypeValue,N}
    #@. val = a / b

    ## loop option
    @simd for k in eachindex(val, a)
        @inbounds val[k] = a[k] / b[1]
    end
    return nothing
end

@doc "scalarDivide(a, b) = a ./ b, where a is a scalar" scalarDivide
@newnode scalarDivide{a,b}::size(b)
@inline function cg_scalarDivide!(
    val::TypeArray{TypeValue,N}, a::TypeArray{TypeValue,0}, b::TypeArray{TypeValue,N}
) where {TypeValue,N}
    #@. val = a / b

    ## loop option
    @simd for k in eachindex(val, b)
        @inbounds val[k] = a[1] / b[k]
    end
    return nothing
end

# overload broadcast of /
"""
a ./ b broadcast maps to pointDivide() or divideScalar() depending on the sizes of the arguments
"""
function Broadcast.broadcasted(
    ::typeof(/),
    graph::ComputationGraph{TypeValue},
    a::Node1,
    b::Node2
) where {TypeValue,Node1<:AbstractNode,Node2<:AbstractNode}
    if size(a) == size(b)
        return pointDivide(graph, a, b)
    elseif size(a) == ()
        return scalarDivide(graph, a, b)
    elseif size(b) == ()
        return divideScalar(graph, a, b)
    else
        display(a)
        display(b)
        error(".* not implements for nodes of these sizes")
    end
end

#########
## Powers
#########

@doc "exponentScalar(a, b) = a .^ b, where b is a scalar" exponentScalar
@newnode exponentScalar{a,b}::size(a)
@inline function cg_exponentScalar!(
    val::TypeArray{TypeValue,N}, a::TypeArray{TypeValue,N}, b::TypeArray{TypeValue,0}
) where {TypeValue,N}
    #@. val = a ^ b

    ## loop option
    @simd for k in eachindex(val, a)
        @inbounds val[k] = a[k]^b[1]
    end
    return nothing
end
# overload usual Base.:^
"""a ^ b maps to exponentScalar(a,b)"""
Base.:^(graph::ComputationGraph{TypeValue}, a::Node1, b::Node2
) where {TypeValue,Node1<:AbstractNode,Node2<:AbstractNode} = exponentScalar(graph, a, b)

#########
## Affine
#########

@doc "affine(A,x,b) = A*x .+ b where b is a vector, x can be a vector or a matrix" affine
@newnode affine{A,x,b}::(length(size(x)) == 1 ? size(b) : (size(A, 1), size(x, 2)))
@inline function cg_affine!(
    val::TypeArray{TypeValue,N},
    A::TypeArray{TypeValue,2},
    x::TypeArray{TypeValue,N},
    b::TypeArray{TypeValue,1},
) where {TypeValue,N}
    ## This option gets about 12.5G flops
    @simd for k in axes(val, 2)
        @inbounds copyto!((@view val[:, k]), b)
    end
    mul!(val, A, x, one(TypeValue), one(TypeValue)) # faster than loop


    #LinearAlgebra.BLAS.gemv!('N', one(TypeValue), A, x, one(TypeValue), val) # only for N=1 (?)

    ## loop option gets about 3G flops
    #@assert size(A) == (length(b), length(x))
    #for (j, xj) in enumerate(x)
    #    @simd for i in axes(A, 1) # simd seems to make relatively little difference 
    #        @inbounds val[i] += A[i, j] * xj # @inbounds roughly doubles the flops
    #    end
    #end
    return nothing
end

@doc "affineRows(A,x,b,rows) = (A*x+b)[rows,:]" affineRows
@newnode affineRows{A,x,b,rows}::(length(size(x)) == 1 ? (length(rows),) : (length(rows), size(x, 2)))
@inline function cg_affineRows!(
    val::TypeArray{TypeValue,N},
    A::TypeArray{TypeValue,2},
    x::TypeArray{TypeValue,N},
    b::TypeArray{TypeValue,N},
    rows::TypeArray{Int,1}
) where {TypeValue,N}
    for (i, r) in enumerate(rows)
        val[i] = b[r] + dot((@view A[r, :]), x)
    end
    return nothing
end

# TODO wasteful to compute A and then through away rows
@doc "selectRows(A,rows) = y, where y[j] =A[rows[j],j]" selectRows
@newnode selectRows{A,rows}::(size(A, 2),)
@inline function cg_selectRows!(
    val::TypeArray{TypeValue,1},
    A::TypeArray{TypeValue,2},
    rows::TypeArray{Int,1}
) where {TypeValue}
    @assert length(rows) == size(A, 2)
    for (i, r) in enumerate(rows)
        val[i] = A[r, i] # risky to use [@simd] & @inbounds
    end
    return nothing
end

# TODO wasteful to do computations with such a sparse matrix
@doc """
    expandColumns(a,rows,nRows) 

Expands a vector a into a matrix A as follows: 
    Given an n-vector a , returns an nRows x n matrix A with 
        A[i,j] = a[j] if i==rows[j] else 0
""" expandColumns
@newnode expandColumns{1,a,rows}::(par1, length(a),)
@inline function cg_expandColumns!(
    val::TypeArray{TypeValue,2},
    a::TypeArray{TypeValue,1},
    rows::TypeArray{Int,1},
    ::Int # nRows not needed since encoded in size
) where {TypeValue}

    fill!(val, zero(TypeValue))
    @assert length(rows) == size(val, 2)
    for (j, rj) in enumerate(rows)
        val[rj, j] = a[j] # risky to use [@simd] & @inbounds
    end
    return nothing
end
export expandColumns
function expandColumns(
    a::TypeArray{TypeValue,1},
    rows::TypeArray{Int,1},
    nRows::Int,
) where {TypeValue}
    val = TypeArray{TypeValue,2}(undef, nRows, length(a))
    cg_expandColumns!(val, a, rows, nRows)
    return val
end

##########
## Maximum
##########

@doc """
    y=findMaxRow(A) 
    
Creates an integer-valued vector y with as many entries as columns of A, where y[j] is equal to the
index of the row of the largest entry in columns j of A.
""" findMaxRow
# cannot use macro since returns int
#@newnode findMaxRow{A}::(size(A, 2),)

""" 
Node of a computation graph used to represent the result of findMaxRow()
"""
struct NodeFindMaxRow{TP<:Tuple,TPI<:Tuple,TV<:AbstractArray,TC} <: ComputationGraphs.AbstractNode
    id::Int
    parameters::TP
    parentIds::TPI
    value::TV
    compute!::TC
end
export findMaxRow
findMaxRow(graph::ComputationGraph{TypeValue}, A::T1,
) where {TypeValue,T1<:AbstractNode} =
    push!(graph, NodeFindMaxRow, cg_findMaxRow!, (), (A.id,), (A.value,), TypeArray{Int}(undef, size(A, 2)), false)
@inline function cg_findMaxRow!(
    val::TypeArray{Int,1},
    A::TypeArray{TypeValue,2},
) where {TypeValue}
    @assert length(val) == size(A, 2)
    for j in axes(A, 2)
        @inbounds val[j] = findmax(@view A[:, j])[2]
    end
    return nothing
end

@doc "maxRow(A) computes a vector y with as many entries as columns of A, where y[j] is equal to the largest entry in columns j of A" maxRow
@newnode maxRow{A}::(size(A, 2),)
@inline function cg_maxRow!(
    val::TypeArray{TypeValue,1},
    A::TypeArray{TypeValue,2},
) where {TypeValue}
    @assert length(val) == size(A, 2)
    for j in axes(A, 2)
        @inbounds val[j] = maximum(@view A[:, j])
    end
    return nothing
end