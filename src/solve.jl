function solve(game::Game, π0::NTuple{2, <:AbstractVector}; tspan=(0.0,100.0))
    p0 = mortar(collect(π0))
    prob = ODEProblem(p0, tspan) do du, u, p, t
        policy_grad!(du, game.m, u)
    end
    return DifferentialEquations.solve(prob, Tsit5())
end

function policy_grad!(∇π, A, π1, π2)
    V1 = dot(π1, A, π2) # value for player 1
    V2 = -V1
    Q1 = mul!(∇π[Block(1)], A, π2)
    Q2 = mul!(∇π[Block(2)], A', π1)
    Q2 .*= -1
    @. ∇π[Block(1)] = π1 * (Q1 - V1)
    @. ∇π[Block(2)] = π2 * (Q2 - V2)
    nothing
end

policy_grad!(∇π, A, p::BlockVector) = policy_grad!(∇π, A, p[Block(1)], p[Block(2)])

function policy_grad(A, π1, π2)
    ∇π = mortar([zeros(length(π1)), zeros(length(π1))]) 
    policy_grad!(∇π, A, π1, π2)
    return ∇π
end
