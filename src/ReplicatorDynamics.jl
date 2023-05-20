module ReplicatorDynamics

using DifferentialEquations
using BlockArrays
using LinearAlgebra

include("game.jl")

include("solve.jl")
export solve, avg_strats

end # module ReplicatorDynamics
