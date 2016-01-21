using POMDPs
using DESPOT
using POMDPToolbox

include("rockSample.jl")
include("rockSampleParticleLB.jl")
include("rockSampleFringeUB.jl")
include("../../upperBound/upperBoundNonStochastic.jl")
include("../../beliefUpdate/beliefUpdateParticle.jl")


function main(;grid_size::Int64 = 4, num_rocks::Int64 = 4, num_reps = 1)

    total_sim_steps                 = 0
    total_discounted_return         = 0.
    total_undiscounted_return       = 0.
    total_run_time                  = 0.
    
    for i in 1:num_reps
        @printf("\n\n\n\n================= Run %d =================\n", i)
        sim_steps,
        discounted_return,
        undiscounted_return,
        run_time = execute(grid_size, num_rocks)
        
        total_sim_steps               += sim_steps
        total_discounted_return       += discounted_return
        total_undiscounted_return     += undiscounted_return
        total_run_time                += run_time
    end
    
    if (num_reps > 1)
        @printf("\n================= Batch Averages =================\n")
        @printf("Number of steps = %d\n", total_sim_steps/num_reps)
        @printf("Discounted return = %.2f\n", total_discounted_return/num_reps)
        @printf("Undiscounted return = %.2f\n", total_undiscounted_return/num_reps)
        @printf("Runtime = %.2f sec\n", total_run_time/num_reps)
    end
end

function execute(grid_size::Int64 = 4, num_rocks::Int64 = 4)

    n_particles = 500 # number of particles to use in the solver and the belief updater
                      # default: 500
    discount = 0.95
    rand_max = 2^31-1 # 2147483647
        
    # generate unique random seeds (optional, if not supplied, default values will be used)
    seed  ::UInt32   = convert(UInt32, 42) # the main random seed that's used to set the other seeds
    w_seed::UInt32   = seed $  n_particles      # world seed, used in the overall simulation
    b_seed::UInt32   = seed $ (n_particles + 1) # belief seed, used for belief particle sampling, among other things
    m_seed::UInt32   = seed $ (n_particles + 2) # model seed, used to initialize the problem model   

    pomdp    = RockSample(grid_size,
                          num_rocks,
                          rand_max = rand_max,      # optional, default: 2^31-1
                          belief_seed = b_seed,     # optional, default: 479
                          model_seed  = m_seed,     # optional, default: 476
                          discount    = discount)   # optional, default: 0.95
    
    # construct a belief updater and specify some of the optional keyword parameters
    bu = DESPOTBeliefUpdater(pomdp::POMDP,
                             seed = seed,
                             rand_max = rand_max,
                             n_particles = n_particles)
                             
    current_belief = create_belief(bu)
    updated_belief = create_belief(bu)
    initial_belief(pomdp, current_belief)
    
    custom_lb   = RockSampleParticleLB(pomdp) # custom lower bound to use with DESPOT solver
    custom_ub   = UpperBoundNonStochastic(pomdp) # custom upper bound to use with DESPOT solver
    
#    solver      = DESPOTSolver{RockSampleState, RockSampleAction}(pomdp,
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
    rewards     = Array(Float64, 0)
    transition_distribution  = POMDPs.create_transition_distribution(pomdp)
    observation_distribution = POMDPs.create_observation_distribution(pomdp)

    # The rest of DESPOT parameters can also be adjusted as shown below.
    # They can also be specified in the DESPOTSolver constructor above, of course.
    
    # Performance tip: control use of computational resources either by 
    # limiting time_per_move, by limiting the number of trials per move, or both.
    # Setting either parameter to 0 or a negative number disables that limit.
    
    solver.config.search_depth = 90 #default: 90
    solver.config.time_per_move = -1 # sec, default: 1, unlimited: -1
    solver.config.pruning_constant = 0
    solver.config.eta = 0.95 # default: 0.95
    solver.config.sim_len = -1 # default: -1
    solver.config.max_trials = 100 # default: -1
    solver.config.approximate_ubound = false
    solver.config.tiny = 1e-6
    solver.config.debug = 0
                              
    rng = DESPOTDefaultRNG(w_seed, rand_max) # used to advance the state of the simulation (world) 
    policy = POMDPs.solve(solver, pomdp)
        
    sim_steps = 0
    println("\nSTARTING STATE: $state")
    show_state(pomdp, state) #TODO: wrap RockSample in a module
    tic() # start the clock
    while !isterminal(pomdp, state) &&
        (solver.config.sim_len == -1 || sim_step < solver.config.sim_len)
        println("\n*************** STEP $(sim_steps+1) ***************")
        action = POMDPs.action(policy, current_belief)
        POMDPs.transition(pomdp, state, action, transition_distribution)
        POMDPs.rand!(rng, next_state, transition_distribution) # update state to next state
        POMDPs.observation(pomdp, state, action, next_state, observation_distribution)
        observation_distribution.debug = 1 #TODO: remove -debug
        POMDPs.rand!(rng, obs, observation_distribution)
        r = POMDPs.reward(pomdp, state, action)
        push!(rewards, r)
        state = deepcopy(next_state) #TODO: see if this is necessary
        POMDPs.update(bu, current_belief, action, obs, updated_belief)
        current_belief = deepcopy(updated_belief) #TODO: perhaps this could be done better
        println("Action = $action")
        println("State = $next_state"); show_state(pomdp, next_state) #TODO: change once abstract types are introduced
        print(  "Observation = "); show_obs(pomdp, obs) #TODO: change once abstract types are introduced
        println("Reward = $r")
        sim_steps += 1
    end
    run_time = toq() # stop the clock
    
    # Compute discounted reward
    discounted_reward = 0
    multiplier = 1
    for r in rewards
        discounted_reward += multiplier * r
        multiplier *= pomdp.discount
    end
    
    println("\n********** EXECUTION SUMMARY **********")    
    @printf("Number of steps = %d\n", sim_steps)
    @printf("Undiscounted return = %.2f\n", sum(rewards))
    @printf("Discounted return = %.2f\n", discounted_reward)
    @printf("Runtime = %.2f sec\n", run_time)
    
    return sim_steps, sum(rewards), discounted_reward, run_time
end
