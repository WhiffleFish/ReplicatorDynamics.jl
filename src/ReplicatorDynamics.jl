module ReplicatorDynamics

using OrdinaryDiffEq
using BlockArrays
using LinearAlgebra

include("game.jl")
export Game

include("solve.jl")
export solve, avg_strats

end # module ReplicatorDynamics
