Base.@kwdef struct RegularizedSolver
    t::Float64 = 100.
    iter::Int = 100
    η::Float64 = 0.2
end

function solve(sol::RegularizedSolver, A::AbstractMatrix, π0::NTuple{2, <:AbstractVector})
    (;iter, η) = sol
    p0 = mortar(collect(π0))
    π_fix = deepcopy(p0)
    for i ∈ 1:iter
        @show i
        prob = OrdinaryDiffEq.ODEProblem(p0, (0.,sol.t)) do du, u, p, t
            reg_policy_grad!(du, A, u, π_fix, η)
        end
        res = OrdinaryDiffEq.solve(prob, Tsit5())
        π_fix = last(res)
    end
    return π_fix
end

function regularized_game(A, π1, π2, π1_fix, π2_fix, η)
    A′ = zero(A)
    @assert simplex_check(π1) "$π1"
    @assert simplex_check(π2) "$π2"
    @assert simplex_check(π1_fix) "$π1_fix"
    @assert simplex_check(π2_fix) "$π2_fix"

    for a1 ∈ eachindex(π1_fix)
        for a2 ∈ eachindex(π2_fix)
            A′[a1, a2] = A[a1, a2] - η*log(π1[a1] / π1_fix[a1]) + η*log(π2[a2] / π2_fix[a2])
        end
    end
    # @show A′
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

reg_policy_grad(A, _π, _π_fix, η) = reg_policy_grad!(zero(_π), A, _π, _π_fix, η)
