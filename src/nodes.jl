
# This type represents an AND-node (Q-node) of the belief tree, branching
# on observations. It maps the set of observations seen during simulations to
# the VNodes that the simulations reach at the next level

  # Fields:
  # obs_to_particles: a mapping from observations seen to the particles (after
  # the transition) that generated the observations. The particles for each
  # observation become the representative set for the corresponding v-node
  # at the next level.
  # depth: depth of the V-node *above* this node.
  # action: action that led to this q-node.
  # first_step_reward: The average first step reward of particles when they
  # took action 'action'.
  # history: history up to the V-node *above* this node.
  # debug: Flag controlling debugging output.

type _QNode{S,A,O,B,T} #Workaround to achieve a circular type definition
    obs_to_particles::Dict{O, Vector{DESPOTParticle{S}}}
    depth::Int64
    action::A
    first_step_reward::Float64
    history::History
    weight_sum::Float64
    obs_to_node::Dict{O,T}
    n_visits::Int64                # Needed for large problems
    bounds::B
  
      # default constructor
      function _QNode(
                    pomdp::POMDP{S,A,O},
                    bounds::B,
                    obs_to_particles::Dict{O,Vector{DESPOTParticle{S}}},
                    depth::Int64,
                    action::A,
                    first_step_reward::Float64,
                    history::History{A,O},
                    config::DESPOTConfig)
                      
            this = new()
            this.obs_to_particles = obs_to_particles
            this.depth = depth
            this.action = action
            this.first_step_reward = first_step_reward
            this.history = history
            this.weight_sum = 0
            this.obs_to_node = Dict{O,T}()
            this.n_visits = 0
            this.bounds = bounds
            
            for (obs, particles) in this.obs_to_particles
                obs_weight_sum = 0.0
                for p in particles
                    obs_weight_sum += p.weight
                end
                this.weight_sum += obs_weight_sum
                add(this.history, action, obs)
                remove_last(this.history)
                this.obs_to_node[obs]   = VNode{S,A,O,B}(
                                          pomdp,
                                          particles,
                                          bounds,
                                          this.depth+1,  # TODO: check depth
                                          obs_weight_sum,
                                          false,
                                          config)
            end
            return this
        end
end

# This type ("Value Node") encapsulates a belief node (and recursively, a
# belief tree). It stores the set of particles associated with the node, an
# AND-node for each action, and some bookkeeping information.

type VNode{S,A,O,B}
  particles::Array{DESPOTParticle{S},1}
  lbound::Float64
  ubound::Float64
  depth::Int64
  default_value::Float64            # Value of the default policy (equals to lbound value
                                    # before any backups are performed)
  pruned_action::A                  # Best action at the node after pruning
  weight::Float64                   # Sum of particle weights at this belief
  best_ub_action::A                 # Action that gives the highest upper bound
  best_lb_action::A                 # Action that gives the highest lower bound
  in_tree::Bool                     # True if the node is visited by trial().
                                    # In order to determine if a node is a fringe node
                                    # of the belief tree, we need to expand it one level.
                                    # The nodes added during this expansion of a fringe
                                    # node are not considered to be within the tree unless
                                    # explicitly visited by trial(), so we use
                                    # this indicator variable.
  n_tree_nodes::Int64               # Number of nodes with in_tree == true in the subtree
                                    # rooted at this node
  q_nodes::Dict{A,_QNode{S,A,O,B,
                VNode{S,A,O,B}}}    # Dict of children q-nodes
  n_visits::Int64                   # Needed for large domains
  n_actions_allowed::Int64          # current number of action branches allowed in the node, needed for large domains
  q_star::Float64                   # best current Q-value, needed for large domains

  # default constructor
  function VNode(
               pomdp::POMDP{S,A,O},
               particles::Vector{DESPOTParticle{S}},
               b::B,
               depth::Int64,
               weight::Float64,
               in_tree::Bool,
               config::DESPOTConfig)

        this = new()
        this.particles          = particles
        this.lbound,
        this.ubound             = bounds(b, pomdp, particles, config)
        this.depth              = depth
        this.default_value      = this.lbound
        this.pruned_action      = A()
        this.weight             = weight
        this.best_ub_action     = A()
        this.best_lb_action     = b.lb.best_action
        this.in_tree            = in_tree
        this.n_tree_nodes       = in_tree ? 1:0
        this.q_nodes            = Dict{A,QNode{S,A,O,B}}()
        this.n_visits           = 0
        this.n_actions_allowed  = 0
        this.q_star             = -Inf
        
        validate_bounds(this.lbound, this.ubound, config)
        return this
  end
end

typealias QNode{S,A,O,B} _QNode{S,A,O,B,VNode{S,A,O,B}}

function get_upper_bound{S,A,O,B}(qnode::QNode{S,A,O,B})
  ubound::Float64 = 0.0
  for (obs, node) in qnode.obs_to_node
      ubound += node.ubound * node.weight
  end
  return ubound/qnode.weight_sum
end

function get_lower_bound{S,A,O,B}(qnode::QNode{S,A,O,B})
  lbound::Float64 = 0.0
  for (obs, node) in qnode.obs_to_node
      lbound += node.lbound * node.weight
  end
  return lbound/qnode.weight_sum
end

#TODO: Fix this

# function prune{S,A,O,L,U}(qnode::QNode{S,A,O,L,U}, total_pruned::Int64, config::DESPOTConfig)
#   cost::Float64 = 0.0
#   total_pruned::Int64 = 0
#   
#   for (obs,node) in qnode.obs_to_node
#     cost, total_pruned += prune(node, total_pruned, config)
#   end
#   return cost, total_pruned
# end

function get_lb_action{S,A,O,B}(node::VNode{S,A,O,B}, config::DESPOTConfig, discount::Float64)
  a_star = A()
  q_star::Float64 = -Inf
  remaining_reward::Float64 = 0.0
  
  for (a,q_node) in node.q_nodes
    remaining_reward = get_lower_bound(q_node)
    if q_node.first_step_reward + discount * remaining_reward > q_star + config.tiny
      q_star = q_node.first_step_reward + discount * remaining_reward
      a_star = a
    end
  end
  return a_star
end

#TODO: fix pruning

# function prune{S,A,O,L,U}(node::VNode{S,A,O,L,U}, total_pruned::Int64, config::DESPOTConfig)
#   # Cost if the node was pruned
#   cost = (config.discount^node.depth) * node.weight * node.default_value
#                 - config.pruning_constant
#   if !node.inTree # Leaf
#     @assert(nodes.n_tree_nodes == 0)
#     return cost, total_pruned
#   end
# 
#   for q_node in node.q_nodes
#     first_step_reward = config.discount^depth * weight * q_node.first_step_reward
#     best_child_value, total_pruned = prune(q_node, total_pruned, config)
# 
#     new_cost = first_step_reward + best_child_value - config.pruning_constant
#     if new_cost > cost
#       cost = new_cost
#       pruned_action = q_node.action
#     end
#   end
#   if pruned_action == -1
#     total_pruned +=1
#   end
#   return cost, total_pruned
# end

