# API

## Contents

```@contents
Pages = ["lib_public.md"]
Depth = 2:3
```

## Public Interface

### Graph creation

```@docs
ComputationGraphs.ComputationGraph
ComputationGraphs.@add

# Variables.jl
variable
constant
zeros
ones
unitvector

# Graph.jl
ComputationGraphs.size
ComputationGraphs.length
ComputationGraphs.typeofvalue
ComputationGraphs.similar
ComputationGraphs.eltype
ComputationGraphs.memory
ComputationGraphs.nodeValue
ComputationGraphs.display
```

### Operations supported

```@docs
# grep @newnode src/LinearAlgebra.jl 
adjoint
adjoint_
adjointTimes
adjointTimesExpandColumns
affine
affineRows
column
LinearAlgebra.dot
divideScalar
expandColumns
expandColumnsTimesAdjoint
exponentScalar
findMaxRow
huber
maxRow
minus
norm1
norm2
plus
pointDivide
pointTimes
scalarDivide
scalarPlus
scalarTimes
selectRows
subtract
sumColumns
sumExpandColumns
times
timesAdjoint
timesAdjointOnes
unitTimesAdjoint
+
-
*
^

# grep @newnode src/Functions.jl 
ComputationGraphs.logistic
ComputationGraphs.relu
ddlogistic
dlogistic
exp
heaviside
sat
sign
sqrt
```

### Differentiation

```@docs
D
hessian
```

### Graph computations

```@docs
set!
compute!
get
copyto!
```

### Recipes

```@docs
gradDescent!
adam!
denseChain!
denseChain
denseQlearningChain
denseChain_FluxZygote
denseChain_FluxEnzyme
```

### Parallelization

```@docs
computeSpawn!
syncValid
ComputationGraphs.request
Base.wait
computeUnspawn!
```

### Code generation

```@docs
Code
sets!
computes!
gets!
copies!
```

## Internal functions

### Graph definition

```@docs
ComputationGraphs.@newnode
Base.push!
ComputationGraphs.nodesAndParents
ComputationGraphs.add2children
ComputationGraphs.children

ComputationGraphs.AbstractNode
ComputationGraphs.AbstractConstantNode
ComputationGraphs.AbstractSpecialNode
ComputationGraphs.noComputation

ComputationGraphs.NodePlus
ComputationGraphs.NodeAdjoint_
ComputationGraphs.NodeAdjointTimes
ComputationGraphs.NodeAdjointTimesExpandColumns
ComputationGraphs.NodeAffine
ComputationGraphs.NodeAffineRows
ComputationGraphs.NodeColumn
ComputationGraphs.NodeConstant
ComputationGraphs.NodeDdlogistic
ComputationGraphs.NodeDivideScalar
ComputationGraphs.NodeDlogistic
ComputationGraphs.NodeDot_
ComputationGraphs.NodeExp_
ComputationGraphs.NodeExpandColumns
ComputationGraphs.NodeExpandColumnsTimesAdjoint
ComputationGraphs.NodeExponentScalar
ComputationGraphs.NodeFindMaxRow
ComputationGraphs.NodeHeaviside
ComputationGraphs.NodeHuber
ComputationGraphs.NodeLogistic_
ComputationGraphs.NodeMaxRow
ComputationGraphs.NodeMinus
ComputationGraphs.NodeNorm1
ComputationGraphs.NodeNorm2
ComputationGraphs.NodeOnes
ComputationGraphs.NodePointDivide
ComputationGraphs.NodePointTimes
ComputationGraphs.NodeRelu
ComputationGraphs.NodeSat
ComputationGraphs.NodeScalarPlus
ComputationGraphs.NodeScalarTimes
ComputationGraphs.NodeSelectRows
ComputationGraphs.NodeScalarDivide
ComputationGraphs.NodeSign_
ComputationGraphs.NodeSqrt_
ComputationGraphs.NodeSubtract
ComputationGraphs.NodeSumColumns
ComputationGraphs.NodeSumExpandColumns
ComputationGraphs.NodeTimes
ComputationGraphs.NodeTimesAdjoint
ComputationGraphs.NodeTimesAdjointOnes
ComputationGraphs.NodeUnitTimesAdjoint
ComputationGraphs.NodeUnitVector
ComputationGraphs.NodeVariable
ComputationGraphs.NodeZeros
```

### Graph evaluation

```@docs
ComputationGraphs.generateComputeFunctions
ComputationGraphs.compute_node!
ComputationGraphs.compute_with_ancestors!
```

### Code generation

```@docs
ComputationGraphs.nodes_str
ComputationGraphs.call_gs
ComputationGraphs.compute_str_recursive
ComputationGraphs.compute_str_unrolled
ComputationGraphs.compute_str_parallel
```

## API index

```@index
Pages = ["lib_public.md"]
```
