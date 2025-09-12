using LogExpFunctions

@doc "sqrt() takes the square root of all entries of a vector or matrix" sqrt
@newnode sqrt_{x}::size(x)
@inline function cg_sqrt_!(val::TV, x::TV) where {TV<:AbstractArray}
    @simd for k in eachindex(x, val)
        @inbounds val[k] = sqrt(x[k])
    end
    return nothing
end
Base.sqrt(graph::ComputationGraph, node::Node) where {Node<:AbstractNode} = sqrt_(graph, node)

@doc "relu() computes the relu (max with 0) of all entries of a vector or matrix" relu
@newnode relu{x}::size(x)
#println(@macroexpand @newnode relu{x}::size(x))
@inline function cg_relu!(val::TV, x::TV) where {TV<:AbstractArray}
    @simd for k in eachindex(x, val)
        @inbounds val[k] = (x[k] > zero(eltype(TV))) ? x[k] : zero(eltype(TV))
    end
    return nothing
end
# "regular" relu (since not available in Base)
relu(x::TV) where {TV<:AbstractArray} =
    [(xi > zero(eltype(TV))) ? xi : zero(eltype(TV)) for xi in x]

@doc "heaviside() computes the heaviside (>0 indicator) of all entries of a vector or matrix" heaviside
@newnode heaviside{x}::size(x)
@inline function cg_heaviside!(val::TV, x::TV) where {TV<:AbstractArray}
    @simd for k in eachindex(x, val)
        @inbounds val[k] = (x[k] > zero(eltype(TV))) ? one(eltype(TV)) : zero(eltype(TV))
    end
    return nothing
end
# "regular" heaviside (since not available in Base)
heaviside(x::TV) where {TV<:AbstractArray} =
    [(xi > zero(eltype(TV))) ? one(eltype(TV)) : zero(eltype(TV)) for xi in x]

@doc "sign() computes the sign function of all entries of a vector or matrix" sign
@newnode sign_{x}::size(x)
@inline function cg_sign_!(val::TV, x::TV) where {TV<:AbstractArray}
    @simd for k in eachindex(x, val)
        @inbounds val[k] = x[k] == zero(eltype(TV)) ? zero(eltype(TV)) :
                           (x[k] > zero(eltype(TV))) ? one(eltype(TV)) : -one(eltype(TV))
    end
    return nothing
end
Base.sign(graph::ComputationGraph, node::Node) where {Node<:AbstractNode} =
    sign_(graph, node)

@doc "sat() computes the saturation function of all entries of a vector or matrix" sat
@newnode sat{x}::size(x)
@inline function cg_sat!(val::TV, x::TV) where {TV<:AbstractArray}
    @simd for k in eachindex(x, val)
        @inbounds val[k] = x[k] >= one(eltype(TV)) ? one(eltype(TV)) :
                           (x[k] <= -one(eltype(TV)) ? -one(eltype(TV)) : x[k])
    end
    return nothing
end

@doc "exp() computes the exponential of all entries of a vector or matrix" exp
@newnode exp_{x}::size(x)
@inline function cg_exp_!(val::TV, x::TV) where {TV<:AbstractArray}
    @simd for k in eachindex(x, val)
        @inbounds val[k] = Base.exp(x[k])
    end
    return nothing
end
Base.exp(graph::ComputationGraph, node::Node) where {Node<:AbstractNode} = exp_(graph, node)

@doc "logistics(x)=1/(1+exp(-x)) computes the logistics function of all entries of a vector or matrix" logistic
@newnode logistic_{x}::size(x)
@inline function cg_logistic_!(val::TV, x::TV) where {TV<:AbstractArray}
    @simd for k in eachindex(x, val)
        @inbounds val[k] = LogExpFunctions.logistic(x[k])
    end
    return nothing
end
LogExpFunctions.logistic(graph::ComputationGraph, node::Node) where {Node<:AbstractNode} =
    logistic_(graph, node)

@doc """
    dlogistics(x)=exp(-x)/(1+exp(-x))^2 

computes the derivative of the logistics function of all entries
of a vector or matrix
""" dlogistic
@newnode dlogistic{x}::size(x)
@inline function cg_dlogistic!(val::TV, x::TV) where {TV<:AbstractArray}
    @simd for k in eachindex(x, val)
        @inbounds ex::eltype(TV) = Base.exp(-x[k])
        @inbounds val[k] = ex / (1 + ex)^2
    end
    return nothing
end

@doc """
    ddlogistic(x)=(exp(-2x)-exp(-x)) /(1+exp(-x))^3 
    
computes the 2nd-derivative of the logistics function of all entries of a vector or matrix
""" ddlogistic
@newnode ddlogistic{x}::size(x)
@inline function cg_ddlogistic!(val::TV, x::TV) where {TV<:AbstractArray}
    @simd for k in eachindex(x, val)
        @inbounds ex::eltype(TV) = Base.exp(-x[k])
        @inbounds val[k] = (ex^2 - ex) / (1 + ex)^3
    end
    return nothing
end