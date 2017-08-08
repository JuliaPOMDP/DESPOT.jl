
mutable struct DESPOTSolver{S,A,O,B,RS} <: POMDPs.Solver
    belief::DESPOTBelief{S,A,O}
    bounds::B
    random_streams::RS
    root::VNode{S,A,O,B}
    node_count::Int64
    config::DESPOTConfig
    rng::AbstractRNG
    #preallocated for simulations
    curr_reward::DESPOTReward
    next_state::S
    curr_obs::O
    print_bounds::Bool

    # default constructor
    function DESPOTSolver{S,A,O,B,RS}(  ;
                            bounds::B = B(), #TODO: fix
                            rng::AbstractRNG = Base.GLOBAL_RNG,
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
                            debug::Int64 = 0,
                            random_streams::RS = RandomStreams(n_particles, search_depth, main_seed),
                            next_state = S(),
                            curr_obs = O(),
                            print_bounds = false
                           ) where {S,A,O,B,RS}

        this = new()

        # supplied variables
        this.bounds = bounds

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
        this.rng = rng
        this.next_state = next_state
        this.curr_obs = curr_obs
        this.curr_reward = 0.0
        this.random_streams = random_streams
        this.print_bounds = print_bounds
        return this
    end
end

function init_solver{S,A,O,B}(solver::DESPOTSolver{S,A,O,B}, pomdp::POMDPs.POMDP{S,A,O})

    # Instantiate random streams

    fill_random_streams!(solver.random_streams, solver.config.rand_max)
    init_bounds(solver.bounds, pomdp, solver.config)

    return nothing
end

function new_root{S,A,O,B}(solver::DESPOTSolver{S,A,O,B},
                  pomdp::POMDP{S,A,O},
                  belief::DESPOTBelief{S})

    solver.belief = belief
    solver.root = VNode{S,A,O,B}(
                        pomdp,
                        belief.particles,
                        solver.bounds,
                        0,
                        1.0,
                        false,
                        solver.config)
    return nothing
end


function search{S,A,O,B}(solver::DESPOTSolver{S,A,O,B}, pomdp::POMDP{S,A,O})
    n_trials::Int64 = 0
    start_time::Float64 = time()
    stop_now::Bool = false

    if solver.print_bounds
        @printf("Before: lBound = %.10f, uBound = %.10f\n", solver.root.lbound, solver.root.ubound)
    end

    while ((excess_uncertainty(solver.root.lbound,
                                solver.root.ubound,
                                solver.root.lbound,
                                solver.root.ubound,
                                0,
                                solver.config.eta,
                                discount(pomdp)) > 1e-6)
                                && !stop_now)

        trial(solver, pomdp, solver.root, n_trials)
        n_trials += 1
        if ((solver.config.max_trials > 0) && (n_trials >= solver.config.max_trials)) ||
        ((solver.config.time_per_move > 0) && ((time() - start_time) >= solver.config.time_per_move))
            stop_now = true
        end
    end

    if solver.print_bounds
        @printf("After:  lBound = %.10f, uBound = %.10f\n", solver.root.lbound, solver.root.ubound)
        @printf("Number of trials: %d\n", n_trials)
    end

    if solver.config.pruning_constant != 0.0
        total_pruned = prune(solver.root, solver.config, discount(pomdp)) # Number of non-child belief nodes pruned
        act = solver.root.pruned_action
        if 
            default_action(solver.bounds, pomdp, solver.root.particles, solver.config)
        return (act == -1 ? default : act), n_trials #TODO: fix actions
    elseif !solver.root.in_tree
        println("Root not in tree")
        default = default_action(solver.bounds, pomdp, solver.root.particles, solver.config)
        return default, n_trials
    else
        return get_lb_action(solver.root, solver.config, discount(pomdp)), n_trials
    end
    return nothing
end

function trial{S,A,O,B}(solver::DESPOTSolver{S,A,O,B}, pomdp::POMDP{S,A,O}, node::VNode{S,A,O,B}, n_trials::Int64)

    n_nodes_added::Int64 = 0
    ubound::Float64 = 0.0

    if (node.depth >= solver.config.search_depth) || isterminal(pomdp, node.particles[1].state)
      return 0 # nodes added
    end

    if isempty(node.q_nodes)
        expand_one_step(solver, pomdp, node, solver.bounds)
    end

    a_star::A = node.best_ub_action

    i_star::Int, weighted_eu_star::Float64 =
                               get_best_weuo(node.q_nodes[a_star],
                                             solver.root,
                                             solver.config,
                                             discount(pomdp)) # it's an array!

    o_star, vnode_star = node.q_nodes[a_star].obs_and_nodes[i_star]

    if weighted_eu_star > 0.0
        add(solver.belief.history, a_star, o_star)
        n_nodes_added = trial(solver,
                            pomdp,
                            vnode_star,
                            n_trials)
        remove_last(solver.belief.history)
    end
    node.n_tree_nodes += n_nodes_added

    # Backup
    potential_lbound = node.q_nodes[a_star].first_step_reward +
                    discount(pomdp) * get_lower_bound(node.q_nodes[a_star])
    node.lbound = max(node.lbound, potential_lbound)

    # As the upper bound of a_star may become smaller than the upper bound of
    # another action, we need to check all actions - unlike the lower bound.
    node.ubound = -Inf

    for a in iterator(actions(pomdp))
        ubound = node.q_nodes[a].first_step_reward +
            discount(pomdp) * get_upper_bound(node.q_nodes[a])
        if ubound > node.ubound
            node.ubound = ubound
            node.best_ub_action = a
        end
    end

    # Sanity check
    if (node.lbound > node.ubound + solver.config.tiny)
        println("depth = $(node.depth)")
        warn("Lower bound ($(node.lbound)) is higher than upper bound ($(node.ubound))")
    end

    if !node.in_tree
        node.in_tree = true
        node.n_tree_nodes += 1
        n_nodes_added += 1
    end
    return n_nodes_added
end

function expand_one_step{S,A,O,B}(solver::DESPOTSolver{S,A,O}, pomdp::POMDP{S,A,O}, node::VNode{S,A,O,B}, bounds::B)

    q_star::Float64 = -Inf
    first_step_reward::Float64 = 0.0
    remaining_reward::Float64 = 0.0
    rng = create_rng(solver.random_streams)

    for curr_action in iterator(actions(pomdp))
        obs_to_particles = Dict{O,Vector{DESPOTParticle{S}}}()

        for p in node.particles

            set_rng_state!(rng, solver.random_streams, p.id+1, node.depth+1)

            step(
                solver,
                pomdp,
                p.state,
                rng,
                curr_action)

            if !haskey(obs_to_particles, solver.curr_obs)
                obs_to_particles[solver.curr_obs] = DESPOTParticle{S}[]
            end

            push!(obs_to_particles[solver.curr_obs],
                  DESPOTParticle(solver.next_state,
                  p.id,
                  p.weight))
            first_step_reward += solver.curr_reward * p.weight
        end

        first_step_reward /= node.weight

        new_qnode = QNode{S,A,O,B}(
                          pomdp,
                          bounds,
                          obs_to_particles,
                          node.depth,
                          curr_action,
                          first_step_reward,
                          solver.belief.history,
                          solver.config)
        node.q_nodes[curr_action] = new_qnode
        remaining_reward = get_upper_bound(new_qnode)

        if (first_step_reward + discount(pomdp)*remaining_reward) > (q_star + solver.config.tiny)
            q_star = first_step_reward + discount(pomdp) * remaining_reward
            node.best_ub_action = curr_action
        end

        first_step_reward = 0.0
    end # for a
    return node
end

# fill in pre-allocated variables
function step{S,A,O,B}(solver::DESPOTSolver{S,A,O,B},
              pomdp::POMDPs.POMDP{S,A,O},
              state::S,
              rng::AbstractRNG,
              action::A)

    solver.next_state, solver.curr_obs, solver.curr_reward =
        POMDPs.generate_sor(pomdp, state, action, rng)

    return nothing
end
