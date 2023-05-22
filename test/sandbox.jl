begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using ReplicatorDynamics
    using BlockArrays
    Pkg.activate(@__DIR__)
end

game = [
        1. -1.
        -1. 1.
]

using Plots
p1 = [0.999, 0.001]
p2 = [0.999, 0.001]

sol = ReplicatorSolver()
res = ReplicatorDynamics.solve(sol, game, (p1,p2))
plot(res)


sol = ReplicatorSolver()
res = ReplicatorDynamics.solve(sol, game, (p1,p2))



using LinearAlgebra

plot(res)

avg = avg_strats(res) 
plot(avg[:,1], avg[:,3])

avg[:,1]

y = cumsum(reduce(hcat, res.u); dims=2)
eachcol(y) ./= axes(y,2)


plot(y[1,:], y[3,:])



game = ReplicatorDynamics.Game(
    Float64[
        0 -1 1
        1 0 -1
        -1 1 0
    ]
)

p1 = [0.90, 0.05, 0.05]
p2 = [0.90, 0.05, 0.05]
ReplicatorDynamics.policy_grad(game.m, p1, p2)
res = ReplicatorDynamics.solve(game,(p1,p2); tspan = (0.0, 1000.))
plot(res, ylims=(0,1))

sum(reduce(hcat, res.u)[1:3,:]; dims=1)
sum(reduce(hcat, res.u)[4:6,:]; dims=1)


avg = avg_strats(res)
plot(avg)
