module ReplicatorDynamics

using DifferentialEquations
using BlockArrays
using LinearAlgebra

include("check.jl")
export simplex_check

include("solve.jl")
export ReplicatorSolver, solve, avg_strats

include("regularized.jl")
export RegularizedSolver, lyapunov

end # module ReplicatorDynamics
