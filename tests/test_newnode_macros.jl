using Revise

using MacroTools

using ComputationGraphs

using Test

@testset "test_newnode_macros: see code" begin
    println(prettify(@macroexpand ComputationGraphs.@newnode name1{}::()))
    println(prettify(@macroexpand ComputationGraphs.@newnode name2{x}::()))
    println(prettify(@macroexpand ComputationGraphs.@newnode name3{x}::(1,)))
    println(prettify(@macroexpand ComputationGraphs.@newnode name4{x}::size(x)))
    println(prettify(@macroexpand ComputationGraphs.@newnode name5{1,x}::size(x)))
    println(prettify(@macroexpand ComputationGraphs.@newnode name6{A,x}::(size(A, 1), size(x, 2))))
end

module test
using ComputationGraphs
ComputationGraphs.@newnode name1{}::()
ComputationGraphs.@newnode name2{x}::()
ComputationGraphs.@newnode name3{x}::(1,)
ComputationGraphs.@newnode name4{x}::size(x)
ComputationGraphs.@newnode name5{1,x}::size(x)
ComputationGraphs.@newnode name6{A,x}::(size(A, 1), size(x, 2))
end

@testset "test_newnode_macros: create module" begin
    using .test

    display(Docs.doc(test.NodeName1))
    display(Docs.doc(test.name1))
    display(Docs.doc(test.NodeName2))
    display(Docs.doc(test.name2))
    display(Docs.doc(test.NodeName3))
    display(Docs.doc(test.name3))
    display(Docs.doc(test.NodeName4))
    display(Docs.doc(test.name4))
    display(Docs.doc(test.NodeName5))
    display(Docs.doc(test.name5))
    display(Docs.doc(test.NodeName6))
    display(Docs.doc(test.name6))
end
