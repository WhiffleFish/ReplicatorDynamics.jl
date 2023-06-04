begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using ReplicatorDynamics
    Pkg.activate(@__DIR__)
    using BlockArrays
    using TernaryPlots
    using Plots
end

game = @SMatrix Float64[
    0 -1 0
    1 0 -1
    -1 1 0
]

p1 = [0.90, 0.05, 0.05]
p2 = [0.90, 0.05, 0.05]

res = ReplicatorDynamics.solve(ReplicatorSolver(t=100.), game, (p1,p2))

## replicator strategy
data = Matrix(reduce(hcat, res(0.0:0.01:100.0).u)[1:3,:]')
t1 = mapreduce(collect∘tern2cart, hcat, eachrow(data))

ax = TernaryPlots.ternary_axes(
    title="Replicator Dynamics",
    xguide="R",
    yguide="P",
    zguide="S",
)
p = Plots.plot!(t1[1,:], t1[2,:], legend=false, lw=2)
