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
+ algebraic simplification of symbolic expression
+ parallelization

A MATLAB implementation of *Computation Graphs* is discussed in

    [1] J. Hespanha.
    TensCalc: A toolbox to generate fast code to solve nonlinear constrained minimizations and compute
    Nash equilibria. Mathematical Programming Computation, 14:451—496, Sep. 2022.
    [pdf](http://www.ece.ucsb.edu/~hespanha/published/tenscalc_journal-svjour3.pdf)

