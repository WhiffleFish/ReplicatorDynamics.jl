using ReplicatorDynamics
using Test

@testset "pennies" begin
    game = Game(
        [
            1. -1.
            -1. 1.
        ]
    )
    p1 = [0.999, 0.001]
    p2 = [0.999, 0.001]
    
    res = solve(game,(p1,p2); tspan = (0.0, 1000.))
    avg = avg_strats(res)
    y = reduce(vcat, res.u')
    @test all(sum(y[:,1:2]; dims=2) .≈ 1.)
    @test all(sum(y[:,3:4]; dims=2) .≈ 1.)

    @test all(sum(avg[:,1:2]; dims=2) .≈ 1.)
    @test all(sum(avg[:,3:4]; dims=2) .≈ 1.)
    @test all(isapprox(avg[end,:], ones(4)*0.5, atol=0.1))
end

@testset "rps" begin
    game = ReplicatorDynamics.Game(
    Float64[
        0 -1 1
        1 0 -1
        -1 1 0
        ]
    )
    p1 = [0.90, 0.05, 0.05]
    p2 = [0.90, 0.05, 0.05]
    
    res = solve(game,(p1,p2); tspan = (0.0, 1000.))
    avg = avg_strats(res)
    y = reduce(vcat, res.u')
    @test all(sum(y[:,1:3]; dims=2) .≈ 1.)
    @test all(sum(y[:,4:6]; dims=2) .≈ 1.)

    @test all(sum(avg[:,1:3]; dims=2) .≈ 1.)
    @test all(sum(avg[:,4:6]; dims=2) .≈ 1.)
    @test all(isapprox(avg[end,:], ones(6)*(1/3), atol=0.1))
end
