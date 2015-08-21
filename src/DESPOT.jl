module DESPOT

using Distributions
using POMDPs
import POMDPs:     
    solve,
    action,
    isterminal,
    update_belief!

typealias DESPOTState       Int64
typealias DESPOTAction      Int64
typealias DESPOTObservation Int64

abstract DESPOTProblem
abstract DESPOTUpperBound
abstract DESPOTLowerBound
abstract DESPOTBeliefUpdate

type DESPOTParticle
  state::Int64
  id::Uint16
  wt::Float32
end

# type DESPOTStateProbability
#   s::Int64
#   p::Float64
# end

include("config.jl")
include("history.jl")
include("randomStreams.jl")
include("world.jl")
include("beliefUpdate/beliefUpdateParticle.jl")

type DESPOTPomdp <: POMDP
    config::DESPOTConfig
    randomStreams::RandomStreams
    problem::DESPOTProblem
    world::World
    history::History
    bu::DESPOTBeliefUpdate

    function DESPOTPomdp (problem::DESPOTProblem;
                            bu::DESPOTBeliefUpdate = DESPOTBeliefUpdateParticle(convert(Uint32,0)),
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
        this.config = DESPOTConfig()
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
        
        # assign default belief updater if none was specified 
        if this.randomStreams.beliefUpdateSeed != 0
            this.bu = DESPOTBeliefUpdateParticle(this.randomStreams.beliefUpdateSeed)
        end
        
        # Instantiate world
        this.world = World (this.problem, this.randomStreams.worldSeed)
        
        # Instantiate history
        this.history = History()
        
        return this
    end
end

include("qnode.jl")
include("vnode.jl")

type DESPOTBelief <: Belief
    particles::Array{DESPOTParticle,1}
    
    function DESPOTBelief (particles::Array{DESPOTParticle,1})
        this = new()
        this.particles = particles
        return this
    end
end

include("solver.jl")
include("utils.jl")

type DESPOTPolicy <: Policy
    solver::DESPOTSolver
    pomdp ::DESPOTPomdp
    
    function DESPOTPolicy(solver::DESPOTSolver, pomdp::DESPOTPomdp)
        this = new()
        this.solver = DESPOTSolver(pomdp)
        this.pomdp = pomdp
        return this
    end
end

# FUNCTIONS
start_state(problem::DESPOTProblem) = error("$(typeof(problem)) does not implement start_state")
initial_belief(problem::DESPOTProblem) = error("$(typeof(problem)) does not implement initial_belief")
lower_bound(problem::DESPOTProblem, particles::Vector{DESPOTParticle}, streamPosition::Int64, config::DESPOTConfig) = 
     error("$(typeof(problem)) does not implement lower_bound")
upper_bound(problem::DESPOTProblem, particles::Vector{DESPOTParticle}) = 
     error("$(typeof(pomdp.problem)) does not implement upper_bound")
init_problem(pomdp::DESPOTPomdp) = error("$(typeof(pomdp.problem)) does not implement init_problem")
step(pomdp::DESPOTPomdp, s::Int64, randNum::Float64, action::Int64) = error("$(typeof(problem)) does not implement step")
display_state(problem::DESPOTProblem, s::Int64) = error("$(typeof(pomdp.problem)) does not implement display_state")
display_obs(problem::DESPOTProblem, obs::Int64) = error("$(typeof(pomdp.problem)) does not implement display_obs")
obs_probability(problem::DESPOTProblem, obs::DESPOTObservation, state::DESPOTState, action::DESPOTAction) =
                                              error("$(typeof(problem)) does not implement obs_probability")
random_state(problem::DESPOTProblem, seed::Uint32) =
                                              error("$(typeof(problem)) does not implement random_state")
# update_belief!(belief::DESPOTBelief, pomdp::DESPOTPomdp, a::DESPOTAction, obs::DESPOTObservation) =
#     error("$(typeof(pomdp.problem)) does not implement belief_update!")
#is_finished(solver::DESPOTSolver, pomdp::DESPOTPomdp) = error("$(typeof(solver)) does not implement is_finished")

import POMDPs: POMDP, Solver, Policy, Belief, action

function action(policy::DESPOTPolicy, belief::DESPOTBelief)
    newRoot (policy.solver, policy.pomdp, belief.particles)
    action, nTrials = search(policy.solver, policy.pomdp)
    return action
end

# function execute_action(pomdp::DESPOTPomdp, action::DESPOTAction)
#     step(pomdp.world, action, pomdp.config)
# end

# Advances the current state of the world
function execute_action(pomdp::DESPOTPomdp, action::Int64)

    if OS_NAME == :Linux
        seed = Cuint[pomdp.world.seed]
        randomNumber = ccall((:rand_r, "libc"), Int, (Ptr{Cuint},), seed) / pomdp.config.randMax
    else #Windows, etc
        srand(pomdp.world.seed)
        randomNumber = rand()
    end
    nextState, reward, obs = step(pomdp.problem, pomdp.world.state, randomNumber, action)
    pomdp.world.state = nextState
    println("Action = $action")
    println("State = $nextState"); display_state(pomdp.problem, nextState)
    print  ("Observation = "); display_obs(pomdp.problem, obs)
    println("Reward = $reward")
    push!(pomdp.world.rewards, reward)
    return obs, reward
end

function update_belief!(belief::DESPOTBelief, pomdp::DESPOTPomdp, action::Int64, obs::Int64)
  belief.particles = run_belief_update(pomdp.bu,
                                       pomdp.problem,
                                       belief.particles,
                                       action,
                                       obs,
                                       pomdp.config)                           
  add(pomdp.history, action, obs)
#  newRoot(solver, problem, particles, config)
end

export
    DESPOTSolver,
    DESPOTPomdp,
    DESPOTPolicy,
    DESPOTProblem,
    DESPOTParticle,
    DESPOTAction,
    DESPOTObservation,
    DESPOTUpperBound,
    DESPOTLowerBound,
    DESPOTBelief,
    DESPOTBeliefUpdate,
    DESPOTConfig,
    solve,
    action,
    execute_action,
    step,
    init_problem,
    start_state,
    initial_belief,
    is_finished,
    lower_bound,
    upper_bound,
    almost_the_same,
    display_state,
    display_obs,
    update_belief!,
    obs_probability,
    discounted_return,
    undiscounted_return,
    random_state
    
# TYPES

function solve(solver::DESPOTSolver, pomdp::DESPOTPomdp)
    policy = DESPOTPolicy (solver, pomdp)
    initSolver(solver, pomdp)
    return policy
end

function isterminal(pomdp::DESPOTPomdp, state::DESPOTState)
    isterminal(pomdp.problem, state)
end

function is_finished(solver::DESPOTSolver, pomdp::DESPOTPomdp)
#  println(solver)
  for p in solver.root.particles
    if !isterminal(pomdp, p.state)
      return false
    end
  end
  return true
end

function undiscounted_return(pomdp::DESPOTPomdp)
    return sum(pomdp.world.rewards)
end

function discounted_return(pomdp::DESPOTPomdp)
    result = 0
    multiplier = 1

    for r in pomdp.world.rewards
        result += multiplier * r
        multiplier *= pomdp.config.discount
    end
    return result
end

end # module

