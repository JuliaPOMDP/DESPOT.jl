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

abstract DESPOTUpperBound
abstract DESPOTLowerBound
abstract DESPOTBeliefUpdate

type DESPOTRandomNumber <: POMDPs.AbstractRNG
    number::Float64
end

type DESPOTParticle{StateType}
  state::StateType
  id::Int64
  weight::Float64
end

#TODO: figure out how to do this properly!
type DESPOTBelief{StateType} <: POMDPs.Belief
    particles::Vector{DESPOTParticle{StateType}}
    history::History 
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
        random_number[1] = rand()
    end
    return nothing
end

include("config.jl")
include("randomStreams.jl")
include("qnode.jl")
include("vnode.jl")
include("utils.jl")
include("solver.jl")

type DESPOTPolicy <: POMDPs.Policy
    solver::DESPOTSolver
    pomdp ::POMDPs.POMDP
end

create_policy(solver::DESPOTSolver, pomdp::POMDPs.POMDP) = DESPOTPolicy(solver, pomdp)

# UPPER and LOWER BOUND FUNCTION INTERFACES
#TODO: try specializing types for DESPOTParticle
lower_bound(lb::DESPOTLowerBound,
            pomdp::POMDPs.POMDP,
            particles::Vector, 
            config::DESPOTConfig) = 
    error("no lower_bound method found for $(typeof(lb)) type")

upper_bound(ub::DESPOTUpperBound,
            pomdp::POMDP,
            particles::Vector, 
            config::DESPOTConfig) = 
    error("no upper_bound method found for $(typeof(lb)) type")
    
init_lower_bound(lb::DESPOTLowerBound,
                    pomdp::POMDPs.POMDP,
                    config::DESPOTConfig) =
    error("$(typeof(lb)) bound does not implement init_lower_bound")                    
    
init_upper_bound(ub::DESPOTUpperBound,
                    pomdp::POMDPs.POMDP,
                    config::DESPOTConfig) =
    error("$(typeof(ub)) bound does not implement init_upper_bound")
    
fringe_upper_bound(pomdp::POMDP, state::POMDPs.State) = 
    error("$(typeof(pomdp)) does not implement fringe_upper_bound")

# FUNCTIONS

function action(policy::DESPOTPolicy, belief::DESPOTBelief)
    new_root(policy.solver, policy.pomdp, belief.particles)
    a, n_trials = search(policy.solver, policy.pomdp) #TODO: return n_trials some other way
    return a
end

function solve(solver::DESPOTSolver, pomdp::POMDPs.POMDP)
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
