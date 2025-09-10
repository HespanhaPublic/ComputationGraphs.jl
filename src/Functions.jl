using LogExpFunctions

@doc "sqrt() takes the square root of all entries of a vector or matrix" sqrt
@newnode sqrt_{x}::size(x)
@inline function cg_sqrt_!(val::TypeArray{TypeValue,N}, x::TypeArray{TypeValue,N}) where {TypeValue,N}
    @simd for k in eachindex(x, val)
        @inbounds val[k] = sqrt(x[k])
    end
    return nothing
end
Base.sqrt(graph::ComputationGraph{TypeValue}, node::Node) where {TypeValue,Node<:AbstractNode} = sqrt_(graph, node)

@doc "relu() computes the relu (max with 0) of all entries of a vector or matrix" relu
@newnode relu{x}::size(x)
#println(@macroexpand @newnode relu{x}::size(x))
@inline function cg_relu!(val::TypeArray{TypeValue,N}, x::TypeArray{TypeValue,N}) where {TypeValue,N}
    @simd for k in eachindex(x, val)
        @inbounds val[k] = (x[k] > zero(TypeValue)) ? x[k] : zero(TypeValue)
    end
    return nothing
end
# "regular" relu (since not available in Base)
relu(x::TypeArray{TypeValue,N}) where {TypeValue,N} =
    [(xi > zero(TypeValue)) ? xi : zero(TypeValue) for xi in x]

@doc "heaviside() computes the heaviside (>0 indicator) of all entries of a vector or matrix" heaviside
@newnode heaviside{x}::size(x)
@inline function cg_heaviside!(val::TypeArray{TypeValue,N}, x::TypeArray{TypeValue,N}
) where {TypeValue,N}
    @simd for k in eachindex(x, val)
        @inbounds val[k] = (x[k] > zero(TypeValue)) ? one(TypeValue) : zero(TypeValue)
    end
    return nothing
end
# "regular" heaviside (since not available in Base)
heaviside(x::TypeArray{TypeValue,N}) where {TypeValue,N} =
    [(xi > zero(TypeValue)) ? one(TypeValue) : zero(TypeValue) for xi in x]

@doc "sign() computes the sign function of all entries of a vector or matrix" sign
@newnode sign_{x}::size(x)
@inline function cg_sign_!(val::TypeArray{TypeValue,N}, x::TypeArray{TypeValue,N}
) where {TypeValue,N}
    @simd for k in eachindex(x, val)
        @inbounds val[k] = x[k] == zero(TypeValue) ? zero(TypeValue) :
                           (x[k] > zero(TypeValue)) ? one(TypeValue) : -one(TypeValue)
    end
    return nothing
end
Base.sign(graph::ComputationGraph{TypeValue}, node::Node) where {TypeValue,Node<:AbstractNode} =
    sign_(graph, node)

@doc "sat() computes the saturation function of all entries of a vector or matrix" sat
@newnode sat{x}::size(x)
@inline function cg_sat!(val::TypeArray{TypeValue,N}, x::TypeArray{TypeValue,N}
) where {TypeValue,N}
    @simd for k in eachindex(x, val)
        @inbounds val[k] = x[k] >= one(TypeValue) ? one(TypeValue) :
                           (x[k] <= -one(TypeValue) ? -one(TypeValue) : x[k])
    end
    return nothing
end

@doc "exp() computes the exponential of all entries of a vector or matrix" exp
@newnode exp_{x}::size(x)
@inline function cg_exp_!(val::TypeArray{TypeValue,N}, x::TypeArray{TypeValue,N}
) where {TypeValue,N}
    @simd for k in eachindex(x, val)
        @inbounds val[k] = Base.exp(x[k])
    end
    return nothing
end
Base.exp(graph::ComputationGraph{TypeValue}, node::Node) where {TypeValue,Node<:AbstractNode} = exp_(graph, node)

@doc "logistics(x)=1/(1+exp(-x)) computes the logistics function of all entries of a vector or matrix" logistic
@newnode logistic_{x}::size(x)
@inline function cg_logistic_!(val::TypeArray{TypeValue,N}, x::TypeArray{TypeValue,N}
) where {TypeValue,N}
    @simd for k in eachindex(x, val)
        @inbounds val[k] = LogExpFunctions.logistic(x[k])
    end
    return nothing
end
LogExpFunctions.logistic(graph::ComputationGraph{TypeValue}, node::Node) where {TypeValue,Node<:AbstractNode} =
    logistic_(graph, node)

@doc """
    dlogistics(x)=exp(-x)/(1+exp(-x))^2 

computes the derivative of the logistics function of all entries
of a vector or matrix
""" dlogistic
@newnode dlogistic{x}::size(x)
@inline function cg_dlogistic!(val::TypeArray{TypeValue,N}, x::TypeArray{TypeValue,N}
) where {TypeValue,N}
    @simd for k in eachindex(x, val)
        @inbounds ex::TypeValue = Base.exp(-x[k])
        @inbounds val[k] = ex / (1 + ex)^2
    end
    return nothing
end

@doc """
    ddlogistic(x)=(exp(-2x)-exp(-x)) /(1+exp(-x))^3 
    
computes the 2nd-derivative of the logistics function of all entries of a vector or matrix
""" ddlogistic
@newnode ddlogistic{x}::size(x)
@inline function cg_ddlogistic!(val::TypeArray{TypeValue,N}, x::TypeArray{TypeValue,N}
) where {TypeValue,N}
    @simd for k in eachindex(x, val)
        @inbounds ex::TypeValue = Base.exp(-x[k])
        @inbounds val[k] = (ex^2 - ex) / (1 + ex)^3
    end
    return nothing
end