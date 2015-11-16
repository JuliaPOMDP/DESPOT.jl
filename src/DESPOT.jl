module DESPOT

using Distributions
using POMDPs

import POMDPs:
        solve,
        action,
        create_policy,
        rand!

include("history.jl")

abstract DESPOTUpperBound
abstract DESPOTLowerBound
abstract DESPOTBeliefUpdate

type DESPOTRandomNumber <: AbstractRNG
    number::Float64
end

type DESPOTParticle{T}
  state::T
  id::Int64
  weight::Float64
end

#TODO: figure out how to do this properly!
type DESPOTBelief{T} <: Belief
    particles::Vector{DESPOTParticle{T}}
    history::History 
end

function rand!(rng::DESPOTRandomNumber)
    return rng.number
end

type DESPOTDefaultRNG <: AbstractRNG
    seed::Array{Uint32,1}
    rand_max::Int64
    debug::Int64
      
    function DESPOTDefaultRNG(seed::Uint32, rand_max::Int64, debug::Int64 = 0)
        this = new()
        this.seed = Cuint[seed]
        this.rand_max = rand_max
        this.debug = debug
        return this
    end    
end

function rand!(rng::DESPOTDefaultRNG)
    if OS_NAME == :Linux
        random_number = ccall((:rand_r, "libc"), Int, (Ptr{Cuint},), rng.seed) / rng.rand_max
    else #Windows, etc
        srand(seed)
        random_number = rand()
    end
    return random_number
end

include("config.jl")
include("randomStreams.jl")
include("qnode.jl")
include("vnode.jl")
include("utils.jl")
include("solver.jl")

type DESPOTPolicy <: Policy
    solver::DESPOTSolver
    pomdp ::POMDP
end

function create_policy(solver::DESPOTSolver, pomdp::POMDP)
    return DESPOTPolicy(solver, pomdp)
end

# UPPER and LOWER BOUND FUNCTION INTERFACES
#TODO: try specializing types for DESPOTParticle
lower_bound(lb::DESPOTLowerBound,
            pomdp::POMDP,
            particles::Vector, 
            config::DESPOTConfig) = 
    error("no lower_bound method found for $(typeof(lb)) type")

upper_bound(ub::DESPOTUpperBound,
            pomdp::POMDP,
            particles::Vector, 
            config::DESPOTConfig) = 
    error("no upper_bound method found for $(typeof(lb)) type")
    
init_lower_bound(lb::DESPOTLowerBound,
                    pomdp::POMDP,
                    config::DESPOTConfig) =
    error("$(typeof(lb)) bound does not implement init_lower_bound")                    
    
init_upper_bound(ub::DESPOTUpperBound,
                    pomdp::POMDP,
                    config::DESPOTConfig) =
    error("$(typeof(ub)) bound does not implement init_upper_bound")
    
fringe_upper_bound(pomdp::POMDP, state::POMDPs.State) = 
    error("$(typeof(pomdp)) does not implement fringe_upper_bound")

# FUNCTIONS

function action(policy::DESPOTPolicy, belief::DESPOTBelief)
    new_root (policy.solver, policy.pomdp, belief.particles)
    a, n_trials = search(policy.solver, policy.pomdp) #TODO: return n_trials some other way
    return a
end

function solve(solver::DESPOTSolver, pomdp::POMDP)
    policy = DESPOTPolicy (solver, pomdp)
    init_solver(solver, pomdp)
    return policy
end

export
    ################## DESPOT TYPES ##################
    DESPOTSolver,
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
