struct CallWrapper <: Function end
(::CallWrapper)(f) = f()

@generated function get_ptr2function(::Type{functionType}) where {functionType}
    quote
        @cfunction($(CallWrapper()), Nothing, ($(Tuple{Ref{functionType}}.parameters[1]),))
    end
end

mutable struct FunctionWrapper
    ptr2function::Ptr{Cvoid}
    # even if the original closure had no arguments, arguments will show up here
    ptr2arguments::Ptr{Cvoid}
    ref2function
    functionType
    function (::Type{FunctionWrapper})(fun::functionType) where {functionType}
        ref2function = Base.cconvert(Ref{functionType}, fun)
        new(get_ptr2function(functionType),
            Base.unsafe_convert(Ref{functionType}, ref2function),
            ref2function,
            functionType)
    end
end

@noinline function reinitializeWrapper(f::FunctionWrapper)
    # get original function
    ref2function = f.ref2function
    functionType = f.functionType
    # reconstruct pointers
    f.ptr2arguments = Base.unsafe_convert(Ref{functionType}, ref2function)
    ptr2function = get_ptr2function(functionType)::Ptr{Cvoid}
    f.ptr2function = ptr2function
    return ptr2function
end

@inline function do_ccall(f::FunctionWrapper)
    ptr2function::Ptr{Cvoid} = f.ptr2function
    if ptr2function == C_NULL
        # For precompile support, only runs rarely and may have allocations
        ptr2function = reinitializeWrapper(f)
        @assert ptr2function != C_NULL
    end
    ptr2arguments::Ptr{Cvoid} = f.ptr2arguments
    ccall(ptr2function, Nothing, (Ptr{Cvoid},), ptr2arguments)
end
#@inline (f::FunctionWrapper)() = do_ccall(f)
