module DESPOT

using Distributions
using POMDPs

import POMDPs:
        solve,
        action,
        create_policy,
        rand,
        rand!

include("history.jl")

abstract DESPOTUpperBound{S,A,O}
abstract DESPOTLowerBound{S,A,O}
abstract DESPOTBeliefUpdate{S,A,O}

type DESPOTRandomNumber <: POMDPs.AbstractRNG
    number::Float64
end

type DESPOTParticle{S}
  state::S
  id::Int64
  weight::Float64
end

type DESPOTBelief{S,A,O} <: POMDPs.Belief{S}
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
    if OS_NAME == :Linux
        random_number[1] = ccall((:rand_r, "libc"), Int, (Ptr{Cuint},), rng.seed) / rng.rand_max
    else #Windows, etc
        srand(rng.seed)
        random_number[1] = Base.rand()
    end
    return nothing
end

include("config.jl")
include("randomStreams.jl")
# include("qnode.jl")
# include("vnode.jl")
include("nodes.jl")
include("utils.jl")
include("solver.jl")

type DESPOTPolicy{S,A,O,L,U} <: POMDPs.Policy{S,A,O}
    solver::DESPOTSolver{S,A,O,L,U}
    pomdp ::POMDPs.POMDP{S,A,O}
end

create_policy{S,A,O,L,U}(solver::DESPOTSolver{S,A,O,L,U}, pomdp::POMDPs.POMDP{S,A,O}) = DESPOTPolicy(solver, pomdp)

# UPPER and LOWER BOUND FUNCTION INTERFACES
lower_bound{S,A,O}(lb::DESPOTLowerBound{S,A,O},
            pomdp::POMDPs.POMDP{S,A,O},
            particles::Vector{DESPOTParticle{S}}, 
            config::DESPOTConfig) = 
    error("no lower_bound method found for $(typeof(lb)) type")

upper_bound{S,A,O}(ub::DESPOTUpperBound{S,A,O},
            pomdp::POMDP{S,A,O},
            particles::Vector{DESPOTParticle{S}},
            config::DESPOTConfig) = 
    error("no upper_bound method found for $(typeof(lb)) type")
    
init_lower_bound{S,A,O}(lb::DESPOTLowerBound{S,A,O},
                    pomdp::POMDPs.POMDP{S,A,O},
                    config::DESPOTConfig) =
    error("$(typeof(lb)) bound does not implement init_lower_bound")                    
    
init_upper_bound{S,A,O}(ub::DESPOTUpperBound{S,A,O},
                    pomdp::POMDPs.POMDP{S,A,O},
                    config::DESPOTConfig) =
    error("$(typeof(ub)) bound does not implement init_upper_bound")
    
fringe_upper_bound{S,A,O}(pomdp::POMDP{S,A,O}, state::S) = 
    error("$(typeof(pomdp)) does not implement fringe_upper_bound")

# FUNCTIONS

function action{S,A,O}(policy::DESPOTPolicy{S,A,O}, belief::DESPOTBelief{S})
    new_root(policy.solver, policy.pomdp, belief.particles)
    a, n_trials = search(policy.solver, policy.pomdp) #TODO: return n_trials some other way
    return a
end

function solve{S,A,O,L,U}(solver::DESPOTSolver{S,A,O,L,U}, pomdp::POMDPs.POMDP{S,A,O})
    policy = DESPOTPolicy(solver, pomdp)
    init_solver(solver, pomdp)
    return policy
end

export
    ################## DESPOT TYPES ##################
    DESPOTSolver,
    DESPOTPolicy,
    DESPOTUpperBound,
    DESPOTLowerBound,
    DESPOTParticle,
    DESPOTBelief,
    DESPOTBeliefUpdate,
    DESPOTConfig,
    DESPOTDefaultRNG,
    DESPOTRandomNumber,
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
    lower_bound,
    upper_bound,
    init_lower_bound,
    init_upper_bound,
    fringe_upper_bound,
    sample_particles! #TODO: need a better way of doing this, perhaps put in POMDPutils
    
end #module
