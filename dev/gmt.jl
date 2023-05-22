begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using ReplicatorDynamics
    Pkg.activate(@__DIR__)
    using BlockArrays
    using GMT
end

game = Float64[
    0 -1 1
    1 0 -1
    -1 1 0
]

p1 = [0.90, 0.05, 0.05]
p2 = [0.90, 0.05, 0.05]

res = ReplicatorDynamics.solve(ReplicatorSolver(t=500.), game, (p1,p2))

## replicator strategy
data = Matrix(reduce(hcat, res(0.0:0.01:20.0).u)[1:3,:]')
t1 = tern2cart(data)
p = ternary(labels=("R", "P", "S"))
GMT.plot!(t1, lw=2, lc=:red)
GMT.text!(tern2cart([0.3 0.4 0.3]), text="Replicator Dynamics", font=18, show=true)

## average strategy
data = Matrix(reduce(hcat, res(0.0:0.01:100.0).u)[1:3,:]')
d2 = avg_strats(data)
t1 = tern2cart(d2)
p = ternary(labels=("R", "P", "S"))
GMT.plot!(t1, lw=1, lc=:red)
GMT.text!([0.5 0.1], text="Avg Replicator Dynamics", font=18, show=true)
