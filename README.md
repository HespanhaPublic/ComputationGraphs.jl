# ComputationGraphs

`ComputationGraphs` is about improving the speed (and energy consumption) of numerical computations
that need to be performed repeatedly, e.g.,
+ one iteration of a numerical optimization algorithm, 
+ one iteration of a filtering/smoothing algorithm,
+ repeated calls to a classification algorithm on different samples, etc.

The computation to be performed is encoded into *Computation Graph* that describes dependencies
between numerical operations and permits several forms of "run-time" optimization, including:
+ allocation-free operation
+ (partial) re-use of perviously performed computations
+ symbolic differentiation
+ symbolic algebraic simplification
+ parallelization

Computation graphs are further discussed in

    [1] J. Hespanha.
    TensCalc: A toolbox to generate fast code to solve nonlinear constrained minimizations and compute
    Nash equilibria. Mathematical Programming Computation, 14:451—496, Sep. 2022.
    [pdf](http://www.ece.ucsb.edu/~hespanha/published/tenscalc_journal-svjour3.pdf)

## To do

### Higher priority

1) try Base.Experimental.@opaque

    https://github.com/JuliaLang/julia/blob/ca2332e868d24115dd8a75b80fde31eb1f9880fe/base/opaque_closure.jl

    DONE: Seems to work, but note clear if makes anything faster.

2) Use LoopVectorization in more LinearAlgebra.jl code

3) add convolution node & differentiation rule & add to NN recipes

4) Make `set!` and regular `copyto!` compatible with parallel computation (synchronize valid bits)

5) Construction of nodes should check sizes (but not cg_...) & possibly make simplification, e.g., sumColumns(vector)=vector

6) Try to understand why in test_NN, an old version (prior to closure) performed faster

7) Try to understand allocations in Example 3 (NN training)

8) Improve documentation: 
    1)  parallel computation
    2)  starting with all the functions in LinearAlgebra, Functions
    3)  train's in NN.jl

9)  creation of computation nodes should take an optional type

10) Support GPU.

    + ComputationTypes should take an optional parameter "TypeArray"

   Some potentially useful references:
   + [GPU](https://www.cise.ufl.edu/~sahni/papers/gpuMatrixMultiply.pdf)
   + [GPU](https://github.com/JuliaGPU/GemmKernels.jl/blob/master/src/kernel.jl)
   + [CPU](https://discourse.julialang.org/t/using-polyester-jl-with-chunksplitters-jl/108528)

11) Add simplifications to avoid adjointTimes, etc.

12) Enlarge the number of operations/functions supported

13) Add softplus(x)=log(1+exp(x))=log(exp(x)(exp(-x)+1))=x+log(1+exp(-x)) , softplus'(x)=logistic(x)

    Fix softplus (& logistic?) to work well when x is very positive or very negative

### Lower priority

1) Improve `@newnode` to shorten [LinearAlgebra](src/LinearAlgebra.jl) & [Functions](src/Functions.jl)

   Perhaps

   1) take the shapes of all inputs and outputs, as in

        ```julia
        @newnode name( (N1,...,N2) , ..., (M1,...M2) ) :: ( K1,...,K2 )`
        ```

      e.g.,

        ```julia
        @newnode norm2( (N) ) :: ()
        @newnode add1( (N) , (N) ) :: (N)
        @newnode add2( (N,M) , (N,M) ) :: (N,M)
        @newnode mul1( (N,M) , (M) ) :: (N)
        @newnode mul2( (N,M) , (M,K) ) :: (N,K)
        @newnode scalarTimes( (), (N) ) :: (N)
        @newnode timesAdjoint( (N), (M) ) :: (N,M)
        ```

2) Use BLAS whenever possible: [BLAS](https://www.netlib.org/blas/blasqr.pdf)

    e.g., `LinearAlgebra.BLAS.axpy!` to do sum and subtraction

    > [!Note]
    >
    > This does not seem to help for operations simpler than matrix multiplication & does not
    > generalize to "variable" size
