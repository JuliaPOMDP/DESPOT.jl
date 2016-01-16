
# type DESPOTSolver{  StateType <: POMDPs.State,
#                     ActionType <: POMDPs.Action,
#                     ObservationType <: POMDPs.Observation} <: POMDPs.Solver
type DESPOTSolver <: POMDPs.Solver
    belief::DESPOTBelief
    lb::DESPOTLowerBound
    ub::DESPOTUpperBound
    random_streams::RandomStreams
    root::VNode
    root_default_action::POMDPs.Action
    node_count::Int64
    config::DESPOTConfig
    #preallocated for simulations
    transition_distribution::POMDPs.AbstractDistribution
    observation_distribution::POMDPs.AbstractDistribution
    rng::DESPOT.DESPOTRandomNumber
    curr_reward::POMDPs.Reward
#     next_state::StateType
#     curr_obs::ObservationType
    next_state::POMDPs.State
    curr_obs::POMDPs.Observation
    StateType::DataType
    ActionType::DataType
    ObservationType::DataType

  # default constructor
#    function DESPOTSolver{StateType, ActionType, ObservationType}(pomdp::POMDPs.POMDP,
    function DESPOTSolver(pomdp::POMDPs.POMDP,
                            belief::DESPOTBelief;
                            lb::DESPOTLowerBound = DESPOTDefaultLowerBound(),
                            ub::DESPOTUpperBound = DESPOTDefaultUpperBound(),
                            search_depth::Int64 = 90,
                            main_seed::UInt32 = convert(UInt32, 42),
                            time_per_move::Float64 = 1.0,                 # sec
                            n_particles::Int64 = 500,
                            pruning_constant::Float64 = 0.0,
                            eta::Float64 = 0.95,
                            sim_len::Int64 = -1,
                            approximate_ubound::Bool = false,
                            tiny::Float64 = 1e-6,
                            max_trials::Int64 = -1,
                            rand_max::Int64 = 2147483647,
                            debug::Int64 = 0
                          )

        this = new()
        
        # supplied variables
        this.belief = belief
        this.lb     = lb
        this.ub     = ub
        
        # Instantiate and initialize config
        this.config = DESPOTConfig()
        this.config.search_depth = search_depth
        this.config.main_seed = main_seed
        this.config.time_per_move = time_per_move
        this.config.n_particles = n_particles
        this.config.pruning_constant = pruning_constant
        this.config.eta = eta
        this.config.sim_len = sim_len
        this.config.approximate_ubound = approximate_ubound
        this.config.tiny = tiny
        this.config.max_trials = max_trials
        this.config.rand_max = rand_max
        this.config.debug = debug
        
        this.root_default_action = create_action(pomdp) # root_default_action
        
        this.rng = DESPOTRandomNumber(-1)
        this.transition_distribution = create_transition_distribution(pomdp)
        this.observation_distribution = create_observation_distribution(pomdp)
        this.next_state = create_state(pomdp)
        this.curr_obs = create_observation(pomdp)
        this.curr_reward = 0.0
        
        this.StateType = typeof(this.next_state)
        this.ActionType = typeof(this.root_default_action)        
        this.ObservationType = typeof(this.curr_obs)
        
        return this
    end
end

function init_solver(solver::DESPOTSolver, pomdp::POMDPs.POMDP)

    # Instantiate random streams
    solver.random_streams = RandomStreams(solver.config.n_particles,
                                          solver.config.search_depth,
                                          solver.config.main_seed)
                                           
    fill_random_streams(solver.random_streams, solver.config.rand_max)
    init_upper_bound(solver.ub, pomdp, solver.config)
    init_lower_bound(solver.lb, pomdp, solver.config)

    return nothing
end

#TODO: Figure out why particles::Vector{Particles} does not work
function new_root{StateType}(solver::DESPOTSolver,
                  pomdp::POMDP,
                  particles::Vector{DESPOTParticle{StateType}})
  
  lbound::Float64, solver.root_default_action = lower_bound(solver.lb,
                                                            pomdp,
                                                            particles,
                                                            solver.ub.upper_bound_act,
                                                            solver.config)
                                                           
  ubound::Float64 = upper_bound(solver.ub,
                                pomdp,
                                particles,
                                solver.config)
                                
  solver.root = VNode{solver.StateType, solver.ActionType}(
                    particles,
                    lbound, 
                    ubound,
                    0,
                    create_action(pomdp), #default(dummy) action
                    1.0,
                    false,
                    solver.config)

  return nothing
end


function search(solver::DESPOTSolver, pomdp::POMDP)
  n_trials = 0
  startTime = time()
  stop_now = false
 
  @printf("Before: lBound = %.10f, uBound = %.10f\n", solver.root.lb, solver.root.ub)
  while ((excess_uncertainty(solver.root.lb,
                             solver.root.ub,
                             solver.root.lb,
                             solver.root.ub,
                             0,
                             solver.config.eta,
                             pomdp.discount) > 1e-6)
                             && !stop_now)

    trial(solver, pomdp, solver.root, n_trials)
    n_trials += 1
    
    if ((solver.config.max_trials > 0) && (n_trials >= solver.config.max_trials)) ||
       ((solver.config.time_per_move > 0) && ((time() - startTime) >= solver.config.time_per_move))
       stop_now = true
    end
  end

  @printf("After:  lBound = %.10f, uBound = %.10f\n", solver.root.lb, solver.root.ub)
  @printf("Number of trials: %d\n", n_trials)

  if solver.config.pruning_constant != 0
    total_pruned = prune(solver.root) # Number of non-child belief nodes pruned
    act = solver.root.prunedAction
    return (act == -1 ? solver.root_default_action : act), currentTrials #TODO: fix actions
  elseif !solver.root.in_tree
      println("Root not in tree")
    return solver.root_default_action, n_trials
  else
    return get_lb_action(solver.root, solver.config, pomdp.discount), n_trials
  end
end

function trial(solver::DESPOTSolver, pomdp::POMDP, node::VNode, n_trials::Int64)
    if (node.depth >= solver.config.search_depth) || isterminal(pomdp, node.particles[1].state)
      return 0 # nodes added
    end
    
    if isempty(node.q_nodes)
        expand_one_step(solver, pomdp, node)
    end

    a_star = node.best_ub_action
    n_nodes_added = 0
    o_star, weighted_eu_star = get_best_weuo(node.q_nodes[a_star], solver.root, solver.config, pomdp.discount) # it's an array!
    println("o_star: $o_star, weighted_eu_star: $weighted_eu_star")
    
    if weighted_eu_star > 0.
        add(solver.belief.history, a_star, o_star)
        n_nodes_added = trial(solver,
                            pomdp,
                            node.q_nodes[a_star].obs_to_node[o_star], # obs_to_node is a Dict
                            n_trials) 
        remove_last(solver.belief.history)
    end
    node.n_tree_nodes += n_nodes_added

    # Backup
    potential_lbound = node.q_nodes[a_star].first_step_reward +
                        pomdp.discount * get_lower_bound(node.q_nodes[a_star])
    node.lb = max(node.lb, potential_lbound)

    # As the upper bound of a_star may become smaller than the upper bound of
    # another action, we need to check all actions - unlike the lower bound.
    node.ub = -Inf

    for a in actions(pomdp)
        ub = node.q_nodes[a].first_step_reward +
              pomdp.discount * get_upper_bound(node.q_nodes[a])
        if ub > node.ub
            node.ub = ub
            node.best_ub_action = a
        end
    end

    # Sanity check
    if (node.lb > node.ub + solver.config.tiny)
        println("depth = $(node.depth)")
        warn("Lower bound ($(node.lb)) is higher than upper bound ($(node.ub))")
    end

    if !node.in_tree
        node.in_tree = true
        node.n_tree_nodes += 1
        n_nodes_added += 1
    end
    return n_nodes_added
end

function expand_one_step(solver::DESPOTSolver, pomdp::POMDP, node::VNode)
  
    q_star::Float64 = -Inf
#    next_state::POMDPs.State = create_state(pomdp)
#    reward::Float64 = 0.0
#    obs::POMDPs.Observation = create_observation(pomdp)
    first_step_reward::Float64 = 0.0
    
    for curr_action in actions(pomdp)
        obs_to_particles = Dict{solver.ObservationType, Vector{DESPOTParticle{solver.StateType}}}()

        for p in node.particles
            step(   
                solver,
                pomdp,
                p.state,
                solver.random_streams.streams[p.id+1, node.depth+1],
                curr_action)
            
            if isterminal(pomdp, solver.next_state) && !isterminal(pomdp, solver.curr_obs)
                error("Terminal state in a particle mismatches observation")
            end

            if !haskey(obs_to_particles, solver.curr_obs)
                obs_to_particles[solver.curr_obs] = DESPOTParticle[]
            end

            push!(obs_to_particles[solver.curr_obs], DESPOTParticle(solver.next_state, p.id, p.weight))
            first_step_reward += solver.curr_reward * p.weight
        end
        
        first_step_reward /= node.weight
        new_qnode = QNode{solver.StateType, solver.ActionType, solver.ObservationType}(
                          pomdp,
                          solver.lb,
                          solver.ub,
                          obs_to_particles,
                          node.depth,
                          curr_action,
                          first_step_reward,
                          solver.belief.history,
                          solver.config)
        node.q_nodes[deepcopy(curr_action)] = new_qnode #TODO: See if this can be done faster
        remaining_reward = get_upper_bound(new_qnode)
        
        if (first_step_reward + pomdp.discount*remaining_reward) > (q_star + solver.config.tiny)
            q_star = first_step_reward + pomdp.discount * remaining_reward
            node.best_ub_action = deepcopy(curr_action) #TODO: See if this can be done better
        end
        
        first_step_reward = 0.0
    end # for a
    return node
end


function step(solver::DESPOTSolver, pomdp::POMDPs.POMDP, state::POMDPs.State, rand_num::Float64, action::POMDPs.Action)
    
    solver.rng.number = rand_num
    POMDPs.transition(pomdp, state, action, solver.transition_distribution)
    POMDPs.rand!(solver.rng, solver.next_state, solver.transition_distribution)
    POMDPs.observation(pomdp, state, action, solver.next_state, solver.observation_distribution)
    POMDPs.rand!(solver.rng, solver.curr_obs, solver.observation_distribution)
    solver.curr_reward = POMDPs.reward(pomdp, state, action)
#    return solver.next_state, solver.reward, solver.observation
    return nothing # fill in pre-allocated variables
end
                                    