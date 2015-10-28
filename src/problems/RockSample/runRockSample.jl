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
    seed = convert(Uint32, 42)
    n_particles = 500 # number of particles to use in the solver and the belief updater
    rand_max = 2147483647
    
    # construct a belief updater and specify some of the optional keyword parameters
    bu = DESPOTBeliefUpdater(pomdp::POMDP,
                             seed = seed,
                             rand_max = rand_max,
                             n_particles = n_particles)
                             
    current_belief = create_belief(bu)
#    println("main 1: n_particles: $(length(current_belief.particles))")
    updated_belief = create_belief(bu)
#    println("main 2: n_particles: $(length(updated_belief.particles))")
    initial_belief(pomdp, current_belief)
#    println("main 3: n_particles: $(length(current_belief.particles))")
    
    custom_lb   = RockSampleParticleLB(pomdp) # custom lower bound to use with DESPOT solver
    custom_ub   = UpperBoundNonStochastic(pomdp) # custom upper bound to use with DESPOT solver
    
    solver      = DESPOTSolver(pomdp,
                               current_belief,
                               # specify some of the optional keyword parameters
                               lb = custom_lb, # use the custom lower bound
                               ub = custom_ub, # use the custom lower bound
                               main_seed = seed, # specify the main random seed
                               n_particles = n_particles)
                               
    state       = POMDPs.create_state(pomdp) # the returned state is also the start state of RockSample
    next_state  = POMDPs.create_state(pomdp)
    obs         = POMDPs.create_observation(pomdp)
    rewards     = Array(Float64,0)
    transition_distribution  = POMDPs.create_transition_distribution(pomdp)
    observation_distribution = POMDPs.create_observation_distribution(pomdp)

    # The rest of DESPOT parameters can also be adjusted as shown below.
    # They can also be specified in the DESPOTSolver constructor above, of course.
    
    # Performance tip: control use of computational resources either by 
    # limiting time_per_move, by limiting the number of trials per move, or both.
    # Setting either parameter to 0 or a negative number disables that limit.
    
    solver.config.search_depth = 90 #default: 90
    solver.config.time_per_move = -1                # sec
    solver.config.pruning_constant = 0
    solver.config.eta = 0.95
    solver.config.sim_len = 3 # default: -1
    solver.config.max_trials = 100 # default: -1
    solver.config.approximate_ubound = false
    solver.config.tiny = 1e-6
    solver.config.debug = 0
                              
    # This call uses predefined seed and rand_max values for consistency
    rng = DESPOTDefaultRNG(convert(Uint32, DESPOT.get_world_seed(solver.random_streams)),
                           solver.config.rand_max)
    policy = POMDPs.solve(solver, pomdp)
        
    sim_step = 0
    println("\nSTARTING STATE:$state")
    show_state(pomdp, state) #TODO: wrap RockSample in a module
    tic() # start the clock
    while !isterminal(pomdp, state) &&
        (solver.config.sim_len == -1 || sim_step < solver.config.sim_len)
        println("\n*************** STEP $(sim_step+1) ***************")
        action = POMDPs.action(policy, current_belief)
        POMDPs.transition(pomdp, state, action, transition_distribution)
        next_state = POMDPs.rand!(rng, next_state, transition_distribution) # update state to next state
        POMDPs.observation(pomdp, state, action, next_state, observation_distribution)
        r = POMDPs.reward(pomdp, state, action)
        push!(rewards, r)
        obs = POMDPs.rand!(rng, obs, observation_distribution)
        state = next_state
# #         println("current belief of length $(length(current_belief.particles)) before: $(current_belief.particles[400:405])")
        POMDPs.update(bu, current_belief, action, obs, updated_belief)
        println("current belief: $(current_belief.particles[1:5])")
        println("updated belief: $(updated_belief.particles[1:5])")
        current_belief = deepcopy(updated_belief) #TODO: perhaps this could be done better
        println("main 4: n_particles: $(length(current_belief.particles))")
        #println("current belief of length $(length(current_belief.particles)) after: $(current_belief.particles[400:405])")
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
