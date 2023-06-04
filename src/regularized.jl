Base.@kwdef struct RegularizedSolver
    t::Float64 = 100.
    iter::Int = 100
    η::Float64 = 0.2
end

struct RegularizedResults{T<:BlockArray}
    res::T
end

function RegularizedResults(data::Vector{<:SciMLBase.ODESolution})
    v = map(data) do res
        reduce(vcat,res.u')
    end
    return mortar(v[:,:])
end

function solve(sol::RegularizedSolver, A::AbstractMatrix, π0::NTuple{2, <:AbstractVector})
    (;iter, η) = sol
    p0 = mortar(collect(π0))
    π_fix = copy(p0)
    res_hist = SciMLBase.ODESolution[]
    for i ∈ 1:iter
        @show i
        prob = OrdinaryDiffEq.ODEProblem(copy(π_fix), (0.,sol.t)) do du, u, p, t
            reg_policy_grad!(du, A, u, π_fix, η)
        end
        res = DifferentialEquations.solve(prob)
        push!(res_hist, res)
        π_fix = last(res)
    end
    return RegularizedResults(res_hist)
end

function regularized_game(A, π1, π2, π1_fix, π2_fix, η)
    A′ = similar(A)
    @assert simplex_check(π1) "$π1"
    @assert simplex_check(π2) "$π2"
    @assert simplex_check(π1_fix) "$π1_fix"
    @assert simplex_check(π2_fix) "$π2_fix"

    for a1 ∈ eachindex(π1_fix)
        for a2 ∈ eachindex(π2_fix)
            A′[a1, a2] = A[a1, a2] - η*log(π1[a1] / π1_fix[a1]) + η*log(π2[a2] / π2_fix[a2])
        end
    end
    return A′
end

function reg_policy_grad!(∇π, A, π1, π2, π1_fix, π2_fix, η)
    A′ = regularized_game(A, π1, π1_fix, π2, π2_fix, η)
    return policy_grad!(∇π, A′, π1, π2)
end


function reg_policy_grad!(∇π, A, _π, _π_fix, η)
    reg_policy_grad!(
        ∇π, A,
        _π[Block(1)], _π[Block(2)], 
        _π_fix[Block(1)], _π_fix[Block(2)],
        η
    )
    return ∇π
end

function lyapunov(Π_fix, Π)
    π1_fix, π2_fix = Π_fix[Block(1)], Π_fix[Block(2)]
    π1, π2 = Π[Block(1)], Π[Block(2)]
    
    v = 0.0
    for a ∈ eachindex(π1)
        v += π1_fix[a]*log(π1_fix[a] / π1[a])
    end
    for a ∈ eachindex(π2)
        v += π2_fix[a]*log(π2_fix[a] / π2[a])
    end
    return v
end

lyapunov(Π_fix::BlockVector) = Base.Fix1(lyapunov, Π_fix)

reg_policy_grad(A, _π, _π_fix, η) = reg_policy_grad!(zero(_π), A, _π, _π_fix, η)
