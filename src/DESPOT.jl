module DESPOT

using Distributions
using POMDPs
using GenerativeModels

import POMDPs:
        solve,
        action,
        initialize_belief,
        update,
        rand,
        rand!

include("history.jl")

abstract DESPOTBeliefUpdate{S,A,O}

typealias DESPOTReward Float64

type DESPOTRandomNumber <: POMDPs.AbstractRNG
    number::Float64
end

type DESPOTParticle{S}
  state::S
  id::Int64
  weight::Float64
end

type DESPOTBelief{S,A,O}
    particles::Vector{DESPOTParticle{S}}
    history::History{A,O}
end

function rand!(rng::DESPOTRandomNumber, random_number::Array{Float64})
    random_number[1] = rng.number
    return nothing
end

type DESPOTDefaultRNG <: POMDPs.AbstractRNG
    seed::Array{UInt32,1}
    rand_max::Int64
    debug::Int64
      
    function DESPOTDefaultRNG(seed::UInt32, rand_max::Int64, debug::Int64 = 0)
        this = new()
        this.seed = Cuint[seed]
        this.rand_max = rand_max
        this.debug = debug
        return this
    end    
end

function rand!(rng::DESPOTDefaultRNG, random_number::Array{Float64})
    if is_linux()
        random_number[1] = ccall((:rand_r, "libc"), Int, (Ptr{Cuint},), rng.seed) / rng.rand_max
    else #Windows, etc
        srand(rng.seed)
        random_number[1] = Base.rand()
    end
    return nothing
end

include("config.jl")
include("randomStreams.jl")
include("nodes.jl")
include("utils.jl")
include("solver.jl")

type DESPOTPolicy{S,A,O,B} <: POMDPs.Policy
    solver::DESPOTSolver{S,A,O,B}
    pomdp ::POMDPs.POMDP{S,A,O}
end

create_policy{S,A,O,B}(solver::DESPOTSolver{S,A,O,B}, pomdp::POMDPs.POMDP{S,A,O}) = DESPOTPolicy(solver, pomdp)

bounds{S,A,O,B}(bounds::B,
            pomdp::POMDP{S,A,O},
            particles::Vector{DESPOTParticle{S}},
            config::DESPOTConfig) = 
    error("no bounds() method found for $(typeof(bounds)) type")
    
init_bounds{S,A,O,B}(bounds::B,
            pomdp::POMDPs.POMDP{S,A,O},
            config::DESPOTConfig) =
    error("$(typeof(bounds)) bound does not implement init_bounds")

# FUNCTIONS

function action{S,A,O}(policy::DESPOTPolicy{S,A,O}, belief::DESPOTBelief{S})
    new_root(policy.solver, policy.pomdp, belief)
    a, n_trials = search(policy.solver, policy.pomdp) #TODO: return n_trials some other way
    return a
end

# for any kind of belief besides DESPOTBelief
function action{S,A,O}(p::DESPOTPolicy{S,A,O}, b)
    N = p.solver.config.n_particles
    pool = Array(DESPOTParticle{S}, N)
    w = 1.0/N
    for i in 1:N
        pool[i] = DESPOTParticle{S}(rand(p.solver.rng, b), i-1, w)
    end

    db = DESPOTBelief(pool, History{A,O}())
    action(p, db)
end

function solve{S,A,O,B}(solver::DESPOTSolver{S,A,O,B}, pomdp::POMDPs.POMDP{S,A,O})
    @warn_requirements solve(solver, pomdp)
    policy = DESPOTPolicy(solver, pomdp)
    init_solver(solver, pomdp)
    return policy
end

@POMDP_require solve(solver::DESPOTSolver, pomdp::POMDP) begin
    P = typeof(pomdp)
    S = state_type(P)
    A = action_type(P)
    O = obs_type(P)
    @req actions(::P)
    @req transition(::P,::S,::A)
    @req observation(::P,::S,::A,::S)    
    @req reward(::P,::S,::A,::S)
    @req discount(::P)    
    @req isterminal(::P,::S)
    @req isterminal_obs(::P,::O)
    as = actions(pomdp)
    @req iterator(::typeof(as))    
end

export
    ################## DESPOT TYPES ##################
    DESPOTSolver,
    DESPOTPolicy,
    DESPOTParticle,
    DESPOTBelief,
    DESPOTBeliefUpdate,
    DESPOTConfig,
    DESPOTDefaultRNG,
    DESPOTRandomNumber,
    DESPOTReward,
    ######## RANDOM STREAMS ######
    RandomStreams,
    MersenneStreamArray,
    ######## HISTORY-RELATED TYPES AND METHODS ######
    History, #TODO: need to handle history-related stuff better, place somewhere else
    add,
    remove_last,
    history_size,
    truncate,
    ############# STANDARD POMDP METHODS ############
    solve,
    action,
    start_state,
    create_policy,
    ########## UPPER AND LOWER BOUND METHODS #########
    bounds,
    init_bounds,
    sample_particles! #TODO: need a better way of doing this, perhaps put in POMDPutils
    
end #module
