module ReplicatorDynamics

using OrdinaryDiffEq
using BlockArrays
using LinearAlgebra

include("check.jl")
export simplex_check

include("solve.jl")
export ReplicatorSolver, solve, avg_strats

include("regularized.jl")
export RegularizedSolver

end # module ReplicatorDynamics
