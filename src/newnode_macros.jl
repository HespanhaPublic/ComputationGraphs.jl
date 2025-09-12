using MacroTools

## Option 1

"""
    @newnode name{C1,...,C2}::outputShape
    @newnode name{Nparameters,C1,...,C2}::outputShape

Macro used to create a new computation node type, where 
+ `C1,...,C2` represent the operands

+ `Nparameters` (optional) represents the number of parameters, which are fixed (as opposed to the
      operands)

+ `outputShape` is the size of the result and 
    + can be a constant Tuple, as in

        ```
        @newnode norm2{x}::()
        ```

    + can use C1,...,C2 (especially their sizes), e.g.,

        ```
        @newnode mult!{A,B}::(size(C1,1),size(C2,2))
        ```

    + can use the values of the parameters, denoted by `par1`, `par2`, ...; as in 

        ```
        @newnode timesAdjointOnes{1,x}::(size(x, 1), par1)
        ```

This macro then generates

    \"\"\" 
    Node of a computation graph used to represent the result of name()
    \"\"\"
    struct NodeName{TP<:Tuple,TPI<:Tuple,TV<:AbstractArray,TC} <: ComputationGraphs.AbstractNode
        id::Int
        parameters::TP
        parentIds::TPI
        value::TV
        compute!::TC
    end

    export name
    name(graph::ComputationGraph,C1::T1,C2::T2,par1,par2,
    ) where {T1<:AbstractNode,T2<:AbstractNode} =
        push!(graph,NodeName,cg_name!,(par1,par2),(C1.id,C2.id),(c1.value,C2.value),outputShape)
"""
macro newnode(expr::Expr)
    @assert @capture(expr, name_{parents__}::shape_) "@newnode name{parents types}::outputShape"

    if !isempty(parents) && isa(parents[1], Int)
        nParameters = parents[1]
        parents = parents[2:end]
    else
        nParameters = 0
    end

    ## Construction of structure code
    nodeName = Symbol("Node", uppercasefirst(string(name)))
    structureType = Expr(
        :curly, nodeName,
        Expr(:(<:), :TP, :Tuple),
        Expr(:(<:), :TPI, :Tuple),
        Expr(:(<:), :TV, Base.AbstractArray),
        :TC
    )
    docString = "Node of a computation graph used to represent the result of " * string(name) * "()"

    ## Construction of push! code

    cg_name! = Symbol("cg_", string(name), "!")

    # parameters type
    parameters = [esc(Symbol("par", i)) for i in 1:nParameters]

    ## build constructor (with types)
    constructorCall = Expr(:call,
        esc(name),                      # constructor name
        Expr(:(::), esc(:graph), :(ComputationGraphs.ComputationGraph)),  # graph
        [Expr(:(::), esc(parents[i]), esc(Symbol("T_", i)))
         for i in 1:length(parents)]...,
        parameters...,                  # parameters 
    )
    #@show constructorCall
    constructorCallWithWhere = Expr(
        :where, constructorCall,
        [Expr(:(<:), esc(Symbol("T_", i)), :(ComputationGraphs.AbstractNode))
         for i in 1:length(parents)]...)
    #@show constructorCallWithWhere

    ## build push! call for constructor
    pushCall = Expr(:call, :push!,
        esc(:graph),                  # graph
        esc(nodeName),                # type
        esc(cg_name!),                # computeFunction
        Expr(:tuple, parameters...),  # parameters array
        Expr(:tuple, [Expr(:., parent, esc(:(:id)))
                      for parent in parents]...), # parents ids
        Expr(:tuple, [Expr(:., parent, esc(:(:value)))
                      for parent in parents]...), # parents values
        esc(shape)                    # output shape
    )
    #@show pushCall

    quote
        # structure declaration
        @doc $(docString) # it might be more informative to fallback to the default documentation for structures, but this would not work for Documenter
        struct $(esc(structureType)) <: $(esc(ComputationGraphs.AbstractNode))
            id::$(esc(Int))
            parameters::$(esc(:TP))
            parentIds::$(esc(:TPI))
            value::$(esc(:TV))
            compute!::$(esc(:TC))
        end
        # constructor
        export $(esc(name))
        #@doc "Constructor for " * $(string(nodeName))
        $(constructorCallWithWhere) = $(pushCall)
    end
end
