module DESPOT

using POMDPs
using Distributions

typealias DESPOTState Int64
abstract DESPOTProblem
abstract DESPOTUpperBound
type DESPOTParticle
  state::Int64
  id::Uint16
  wt::Float32
end

type DESPOTStateProbability
  s::Int64
  p::Float64
end

#include("types.jl")
include("config.jl")
include("history.jl")
include("randomStreams.jl")
#include("problems/RockSample/rockSample.jl")
include("lowerBound/lowerBound.jl")
include("beliefUpdate/beliefUpdate.jl")
#include("beliefUpdate/beliefUpdateParticle.jl")
include("qnode.jl")
include("vnode.jl")
include("solver.jl")
include("world.jl")
include("utils.jl")

import POMDPs: POMDP, Solver, Policy, Belief, action

export
    DESPOTSolver,
    DESPOTPomdp,
    DESPOTPolicy,
    DESPOTProblem,
    DESPOTParticle,
    DESPOTStateProbability,
    DESPOTUpperBound,
    solve,
    action

# TYPES



type DESPOTBelief <: Belief
    particles::Array{DESPOTParticle,1}
end

type DESPOTPomdp <: POMDP
    config::Config
    randomStreams::RandomStreams
    problem::DESPOTProblem
    world::World
    history::History

    function DESPOTPomdp (problem::DESPOTProblem;
                            searchDepth::Int64 = 90,
                            discount::Float64 = 0.95,
                            rootSeed::Int64 = 42,
                            timePerMove::Float64 = 1.,                 # sec
                            nParticles::Int64 = 500,
                            pruningConstant::Float64 = 0.,
                            eta::Float64 = 0.95,
                            simLen::Int64 = -1,
                            approximateUBound::Bool = false,
                            particleWtThreshold::Float64 = 1e-20,
                            numEffParticleFraction::Float64 = 0.05,
                            tiny::Float64 = 1e-6,
                            maxTrials::Int64 = -1,
                            randMax::Int64 = 2147483647,
                            debug::Int64 = 0
                          )
        this = new()
        
        # Assign problem
        this.problem = problem
        
        # Instantiate and initialize config
        this.config = Config()
        this.config.searchDepth = searchDepth
        this.config.discount = discount
        this.config.rootSeed = rootSeed
        this.config.timePerMove = timePerMove
        this.config.nParticles = nParticles
        this.config.pruningConstant = pruningConstant
        this.config.eta = eta
        this.config.simLen = simLen
        this.config.approximateUBound = approximateUBound
        this.config.particleWtThreshold = particleWtThreshold
        this.config.numEffParticleFraction = numEffParticleFraction
        this.config.tiny = tiny
        this.config.maxTrials = maxTrials
        this.config.randMax = randMax
        this.config.debug = debug
        
        # Instantiate random streams
        this.randomStreams = RandomStreams(this.config.nParticles,
                                           this.config.searchDepth,
                                           this.config.rootSeed)
        
        # Instantiate world
        this.world = World (this.problem, getWorldSeed(this.randomStreams))
        
        # Instantiate history
        this.history = History()
    end
end

type DESPOTPolicy <: Policy
    solver::DESPOTSolver
    pomdp ::DESPOTPomdp
    bu::DESPOTBeliefUpdate
    
    function DESPOTPolicy(solver::DESPOTSolver, pomdp::DESPOTPomdp)
        this = new()
        this.bu = BeliefUpdate(getBeliefUpdateSeed(pomdp.randomStreams))
        this.solver = DESPOTSolver(DESPOTBelief(initialBelief), this.bu, pomdp.randomStreams)
        this.pomdp = pomdp
    end
end

# FUNCTIONS
function start_state(problem::DESPOTProblem) = error("$(typeof(problem)) does not implement start_state")

function solve (solver::DESPOTSolver, pomdp::DESPOTPomdp)
    policy = DESPOTPolicy (solver, pomdp)
    return policy
end

function action(policy::DESPOTPolicy, belief::Belief)
    newRoot (policy.solver, policy.pomdp.problem, belief.particles, policy.pomdp.config)
    action, nTrials = search(policy.solver, policy.pomdp.problem, policy.pomdp.config)
    return action
end

function isterminal(pomdp::DESPOTPomdp, state::DESPOTState = nothing)
    # current belief state is taken from the POMDP structure
    return finished(pomdp.solver, pomdp.solver)
end

end # module

