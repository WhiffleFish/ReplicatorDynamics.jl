module ReplicatorDynamics

using DifferentialEquations
using BlockArrays
using LinearAlgebra

include("game.jl")

include("solve.jl")

end # module ReplicatorDynamics
