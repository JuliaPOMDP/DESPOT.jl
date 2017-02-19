using POMDPs
using DESPOT
using POMDPToolbox

include("rockSample.jl")
include("rockSampleParticleLB.jl")
include("rockSampleFringeUB.jl")
include("../../upperBound/upperBoundNonStochastic.jl")
include("rockSampleBounds.jl")
include("../../beliefUpdate/beliefUpdateParticle.jl")

function main(;
                grid_size::Int64            = 4,
                num_rocks::Int64            = 4,
                n_reps::Int64               = 1,
                n_particles::Int64          = 500, # for both solver and belief updater
                main_seed::Int64            = 42,
                discount::Float64           = 0.95,
                search_depth::Int64         = 90,
                time_per_move::Float64      = 1.0,
                pruning_constant::Float64   = 0.0,
                eta::Float64                = 0.95,
                sim_len::Int64              = -1,
                max_trials::Int64           = -1,
                approximate_ubound::Bool    = false,
                debug::Int64                = 0
                )

    total_sim_steps::Int64                  = 0
    total_discounted_return::Float64        = 0.
    total_undiscounted_return::Float64      = 0.
    total_run_time::Float64                 = 0.
    
    # Optional parameters can be adjusted, as shown below.
    # Performance tip: control use of computational resources either by 
    # limiting time_per_move, by limiting the number of trials per move, or both.
    # Setting either parameter to 0 or a negative number disables that limit.
    
    search_depth::Int64 = 90 #default: 90
    time_per_move::Float64 = -1.0 # sec, default: 1, unlimited: -1
    pruning_constant::Float64 = 0.0
    eta::Float64 = 0.95 # default: 0.95
    sim_len::Int64 = -1 # default: -1
    max_trials::Int64 = 100 # default: -1
    approximate_ubound::Bool = false
    tiny::Float64 = 1e-6
    debug::Int64 = 0
    
    for i in 1:n_reps
        @printf("\n\n\n\n================= Run %d =================\n", i)
        sim_steps,
        discounted_return,
        undiscounted_return,
        run_time = execute(
                    grid_size = grid_size,
                    num_rocks = num_rocks,
                    n_particles = n_particles,
                    main_seed = main_seed,
                    discount = discount,
                    search_depth = search_depth,
                    time_per_move = time_per_move,
                    pruning_constant = pruning_constant, 
                    eta = eta,
                    sim_len = sim_len,
                    max_trials = max_trials,
                    approximate_ubound = approximate_ubound,
                    debug = debug
                    )
        
        total_sim_steps               += sim_steps
        total_discounted_return       += discounted_return
        total_undiscounted_return     += undiscounted_return
        total_run_time                += run_time
    end
    
    avg_sim_steps               = total_sim_steps/n_reps
    avg_discounted_return       = total_discounted_return/n_reps
    avg_undiscounted_return     = total_undiscounted_return/n_reps
    avg_run_time                = total_run_time/n_reps
    
    if (n_reps > 1)
        @printf("\n================= Batch Averages =================\n")
        @printf("Number of steps = %d\n", avg_sim_steps)
        @printf("Discounted return = %.2f\n", avg_discounted_return)
        @printf("Undiscounted return = %.2f\n", avg_undiscounted_return)
        @printf("Runtime = %.2f sec\n", avg_run_time)
    end
    return avg_sim_steps, avg_discounted_return, avg_undiscounted_return, avg_run_time
end

function execute(;
                grid_size::Int64            = 4,
                num_rocks::Int64            = 4,
                n_particles::Int64          = 500, # for both solver and belief updater
                main_seed::Int64            = 42,
                discount::Float64           = 0.95,
                search_depth::Int64         = 90,
                time_per_move::Float64      = 1.0,
                pruning_constant::Float64   = 0.0,
                eta::Float64                = 0.95,
                sim_len::Int64              = -1,
                max_trials::Int64           = -1,
                approximate_ubound::Bool    = false,
                debug::Int64                = 0
                )

    rand_max::Int64 = 2^31-1 # 2147483647
        
    # generate unique random seeds (optional, if not supplied, default values will be used)
    seed  ::UInt32   = convert(UInt32, main_seed)  # the main random seed that's used to set the other seeds
    w_seed::UInt32   = seed $  n_particles      # world seed, used in the overall simulation
    b_seed::UInt32   = seed $ (n_particles + 1) # belief seed, used for belief particle sampling, among other things
    m_seed::UInt32   = seed $ (n_particles + 2) # model seed, used to initialize the problem model   

    pomdp   = RockSample(
                        grid_size,
                        num_rocks,
                        rand_max = rand_max,      # optional, default: 2^31-1
                        belief_seed = b_seed,     # optional, default: 479
                        model_seed  = m_seed,     # optional, default: 476
                        discount    = discount)   # optional, default: 0.95
    
    # construct a belief updater and specify some of the optional keyword parameters
    bu = DESPOTBeliefUpdater{RockSampleState,
                             RockSampleAction,
                             RockSampleObs}(
                             pomdp::POMDP,
                             seed = seed,
                             rand_max = rand_max,
                             n_particles = n_particles)
    
    # create initial belief and allocate an updated belief object
    initial_states = initial_state_distribution(pomdp)
    current_belief = initialize_belief(bu, initial_states) 
    updated_belief = create_belief(bu)
    custom_bounds = RockSampleBounds(pomdp)
      
    solver::DESPOTSolver = DESPOTSolver{RockSampleState,
                                        RockSampleAction,
                                        RockSampleObs,
                                        RockSampleBounds}(
                               # specify the optional keyword parameters
                               bounds = custom_bounds,
                               search_depth = search_depth,                                                                                       
                               main_seed = seed, # specify the main random seed
                               time_per_move = time_per_move,
                               n_particles = n_particles,
                               pruning_constant = pruning_constant,
                               eta = eta,
                               sim_len = sim_len,
                               approximate_ubound = approximate_ubound,
                               max_trials =  max_trials,
                               rand_max = rand_max,
                               debug = debug)
                               
    state::RockSampleState       = start_state(pomdp)
    next_state::RockSampleState  = RockSampleState()
    obs::RockSampleObs   = RockSampleObs()
    rewards::Array{Float64}      = Array(Float64, 0)
                                  
    rng::DESPOTDefaultRNG = DESPOTDefaultRNG(w_seed, rand_max) # used to advance the state of the simulation (world) 
    policy::DESPOTPolicy = POMDPs.solve(solver, pomdp)
        
    sim_steps::Int64 = 0
    r::Float64 = 0.0
    println("\nSTARTING STATE: $state")
    show_state(pomdp, state) #TODO: wrap RockSample in a module
    tic() # start the clock
    while !isterminal(pomdp, state) &&
        (solver.config.sim_len == -1 || sim_steps < solver.config.sim_len)
        println("\n*************** STEP $(sim_steps+1) ***************")
        action = POMDPs.action(policy, current_belief)
        transition_distribution = POMDPs.transition(pomdp, state, action)
        next_state = POMDPs.rand(rng, transition_distribution, next_state) # update state to next state
        observation_distribution = POMDPs.observation(pomdp, state, action, next_state)
        observation_distribution.debug = 1 #TODO: remove -debug
        obs = POMDPs.rand(rng, observation_distribution)
        r = POMDPs.reward(pomdp, state, action)
        push!(rewards, r)
        state = next_state        
        updated_belief = POMDPs.update(bu, current_belief, action, obs)
        current_belief = deepcopy(updated_belief) #TODO: perhaps this could be done better
        println("Action = $action")
        println("State = $next_state"); show_state(pomdp, next_state) #TODO: change once abstract types are introduced
        print(  "Observation = "); show_obs(pomdp, obs) #TODO: change once abstract types are introduced
        println("Reward = $r")
        sim_steps += 1
    end
    run_time::Float64 = toq() # stop the clock
    
    # Compute discounted reward
    discounted_reward::Float64 = 0.0
    multiplier::Float64 = 1.0
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
