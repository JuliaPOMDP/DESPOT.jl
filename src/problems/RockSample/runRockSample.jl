using POMDPs
using DESPOT
using Types

include("rockSample.jl")
include("rockSampleParticleLB.jl")
include("rockSampleFringeUB.jl")
include("../../upperBound/upperBoundNonStochastic.jl")
include("../../beliefUpdate/beliefUpdateParticle.jl")

function main(;grid_size::Int64 = 4, num_rocks::Int64 = 4)

    pomdp       = RockSample(grid_size, num_rocks)
    custom_lb   = RockSampleParticleLB(pomdp) # custom lower bound to use with DESPOT solver
    custom_ub   = UpperBoundNonStochastic(pomdp) # custom upper bound to use with DESPOT solver
    current_belief = initial_belief(pomdp)  # pomdp's initial belief
    updated_belief = create_belief(pomdp)
    solver      = DESPOTSolver(pomdp,
                               current_belief,
                               lb = custom_lb, # use the custom lower bound
                               ub = custom_ub) # use the custom lower bound
    state       = POMDPs.create_state(pomdp) # the returned state is also the start state of RockSample
    next_state  = POMDPs.create_state(pomdp)
    obs         = POMDPs.create_observation(pomdp)
    rewards     = Array(Float64,0)
    transition_distribution  = POMDPs.create_transition_distribution(pomdp)
    observation_distribution = POMDPs.create_observation_distribution(pomdp)

    # Here is how you can adjust the default DESPOT parameters, if they were not passed
    # through the optional arguments of the DESPOTSolver constructor above (if desired).
    
    # control use of computational resources either by limiting time_per_move
    # or by limiting the number of trials per move (or both). Setting either
    # to 0 or a negative number disables that limit.
    
    solver.config.search_depth = 90
    solver.config.root_seed = 42
    solver.config.time_per_move = 10                 # sec
    solver.config.n_particles = 500
    solver.config.pruning_constant = 0
    solver.config.eta = 0.95
    solver.config.sim_len = -1
    solver.config.approximate_ubound = false
#     solver.config.particle_weight_threshold = 1e-20
#     solver.config.eff_particle_fraction = 0.05
    solver.config.tiny = 1e-6
    solver.config.max_trials = -1 # default: -1
    solver.config.rand_max = 2147483647
    solver.config.debug = 0
    
    # construct a belief updater and use the same values for parameters as the solver,
    # wherever appropriate
    bu = ParticleBeliefUpdater(pomdp::POMDP,
                               convert(Uint32, DESPOT.get_belief_update_seed(solver.random_streams)),
                               solver.config.rand_max,
                               1e-20,
                               0.05)
                              
    # This call uses predefined seed and rand_max values for consistency
    rng = DESPOTDefaultRNG(convert(Uint32, DESPOT.get_world_seed(solver.random_streams)),
                           solver.config.rand_max)
    policy = POMDPs.solve(solver, pomdp)
        
    sim_step = 0
    println("\nSTARTING STATE:$state")
    show_state(pomdp, state) #TODO: wrap RockSample in a module
    tic() # start the clock
    while !isterminal(pomdp, state) &&
        (solver.config.sim_len == -1 || simStep < solver.config.sim_len)
        println("\n*************** STEP $(sim_step+1) ***************")
        action = POMDPs.action(policy, current_belief)
        POMDPs.transition(pomdp, state, action, transition_distribution)
        next_state = POMDPs.rand!(rng, next_state, transition_distribution) # update state to next state
        POMDPs.observation(pomdp, next_state, action, observation_distribution)
        r = POMDPs.reward(pomdp, state, action)
        push!(rewards, r)
        obs = POMDPs.rand!(rng, obs, observation_distribution)
        state = next_state
        println("current belief of length $(length(current_belief.particles)) before: $(current_belief.particles[400:405])")
        POMDPs.belief(bu, pomdp, current_belief, action, obs, updated_belief)
        current_belief = deepcopy(updated_belief) #TODO: perhaps this could be done better
        println("current belief of length $(length(current_belief.particles)) after: $(current_belief.particles[400:405])")
        println("Action = $action")
        println("State = $next_state"); show_state(pomdp, next_state) #TODO: change once abstract types are introduced
        print  ("Observation = "); show_obs(pomdp, obs) #TODO: change once abstract types are introduced
        println("Reward = $r")
        sim_step += 1
    end
    run_time = toq() # stop the clock
    
    @printf("Number of steps = %d\n", sim_step)
    @printf("Discounted return = %.2f\n", sum(rewards))
    @printf("Runtime = %.2f sec\n", run_time)
end
