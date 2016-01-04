
# This type represents an AND-node (Q-node) of the belief tree, branching
# on observations. It maps the set of observations seen during simulations to
# the VNodes that the simulations reach at the next level

  # Fields:
  # obs_to_particles: a mapping from observations seen to the particles (after
  # the transition) that generated the observations. The particles for each
  # observation become the representative set for the corresponding v-node
  # at the next level.
  # depth: depth of the v-node *above* this node.
  # action: action that led to this q-node.
  # first_step_reward: The average first step reward of particles when they
  # took action 'action'.
  # history: history up to the v-node *above* this node.
  # debug: Flag controlling debugging output.


type QNode{StateType, ActionType, ObservationType}
  obs_to_particles::Dict{ObservationType, Vector{DESPOTParticle{StateType}}}
  depth::Int64
  action::POMDPs.Action
  first_step_reward::Float64
  history::History
  weight_sum::Float64
  obs_to_node::Dict #TODO: See if this can be specified better
  n_visits::Int64                # Needed for large problems
  lb::DESPOTLowerBound
  ub::DESPOTUpperBound
  
      # default constructor
      function QNode{StateType, ActionType, ObservationType}(
                    pomdp::POMDP,
                    lb::DESPOTLowerBound,
                    ub::DESPOTUpperBound,
                    obs_to_particles::Dict{ObservationType, Vector{DESPOTParticle{StateType}}},
                    depth::Int64,
                    action::ActionType,
                    first_step_reward::Float64,
                    history::History{ActionType, ObservationType},
                    config::DESPOTConfig)
                      
            this = new()
            this.obs_to_particles = obs_to_particles
            this.depth = depth
            this.action = action
            this.first_step_reward = first_step_reward
            this.history = history
            this.weight_sum = 0
            this.obs_to_node = Dict{ObservationType, VNode{StateType, ActionType}}()
            this.n_visits = 0
            this.lb = lb
            this.ub = ub
            
            for (obs, particles) in this.obs_to_particles
                obs_weight_sum = 0.0
                for p in particles
                    obs_weight_sum += p.weight
                end
                this.weight_sum += obs_weight_sum

                add(this.history, action, obs)
                l::Float64, action::ActionType = DESPOT.lower_bound(
                                                    lb,
                                                    pomdp,
                                                    particles,
                                                    ub.upper_bound_act,
                                                    config)
                u::Float64 = upper_bound(ub, pomdp, particles, config)
                remove_last(this.history)
                this.obs_to_node[obs] = VNode{StateType, ActionType}(
                                            particles,
                                            l,
                                            u,
                                            this.depth+1,  # TODO: check depth
                                            create_action(pomdp),
                                            obs_weight_sum,
                                            false,
                                            config)
            end
            return this
        end
end

function get_upper_bound(qnode::QNode)
  ub = 0.
  for (obs, node) in qnode.obs_to_node
      ub += node.ub * node.weight
  end
  return ub/qnode.weight_sum
end

function get_lower_bound(qnode::QNode)
  lb = 0.
  for (obs, node) in qnode.obs_to_node
      lb += node.lb * node.weight
  end
  return lb/qnode.weight_sum
end

#TODO: Fix this
function prune(qnode::QNode, total_pruned::Int64, config::DESPOTConfig)
  cost = 0.
  for (obs,node) in qnode.obs_to_node
    cost, total_pruned += prune(node, total_pruned, config)
  end
  return cost, total_pruned
end
