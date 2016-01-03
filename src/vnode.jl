
# This class ("Value Node") encapsulates a belief node (and recursively, a
# belief tree). It stores the set of particles associated with the node, an
# AND-node for each action, and some bookkeeping information.

type VNode{StateType, ActionType}
  particles::Array{DESPOTParticle{StateType},1}
  lb::Float64
  ub::Float64
  depth::Int64
  default_value::Float64            # Value of the default policy (= lbound value
                                    # before any backups are performed)
  pruned_action::ActionType         # Best action at the node after pruning
  weight::Float64                   # Sum of particle weights at this belief
  best_ub_action::ActionType        # Action that gives the highest upper bound
  in_tree::Bool                     # True if the node is visited by Solver::trial().
                                    # In order to determine if a node is a fringe node
                                    # of the belief tree, we need to expand it one level.
                                    # The nodes added during this expansion of a fringe
                                    # node are not considered to be within the tree unless
                                    # explicitly visited by Solver::Trial(), so we use
                                    # this indicator variable.
  n_tree_nodes::Int64               # Number of nodes with inTree == true in the subtree
                                    # rooted at this node
  q_nodes::Dict{ActionType, QNode}  # Dict of children q-nodes
  n_visits::Int64                   # Needed for large domains
  n_actions_allowed::Int64          # current number of action branches allowed in the node, needed for large domains
  q_star::Float64                   # best current Q-value, needed for large domains
  

  # default constructor
  function VNode{StateType, ActionType}( 
               particles::Vector{DESPOTParticle{StateType}},
               l_bound::Float64,
               u_bound::Float64,
               depth::Int64,
               default_action::ActionType,
               weight::Float64,
               in_tree::Bool,
               config::DESPOTConfig)

        this = new()
        this.particles          = particles
        this.lb                 = l_bound
        this.ub                 = u_bound
        this.depth              = depth
        this.default_value      = l_bound
        this.pruned_action      = default_action
        this.weight             = weight
        this.best_ub_action     = default_action
        this.in_tree            = in_tree
        this.n_tree_nodes       = in_tree ? 1:0
        this.q_nodes            = Dict{ActionType,QNode}()
        this.n_visits           = 0
        this.n_actions_allowed  = 0
        this.q_star             = -Inf
        
        validate_bounds(l_bound, u_bound, config)
        return this
  end
end

function get_lb_action(node::VNode, config::DESPOTConfig, discount::Float64)
  a_star = -1
  q_star = -Inf
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
function prune(node::VNode, total_pruned::Int64, config::DESPOTConfig)
  # Cost if the node were pruned
  cost = (config.discount^node.depth) * node.weight * node.default_value
                - config.pruning_constant

  if !node.inTree # Leaf
    @assert(nodes.n_tree_nodes == 0)
    return cost, total_pruned
  end

  for q_node in node.q_nodes
    first_step_reward = config.discount^depth * weight * q_node.first_step_reward
    best_child_value, total_pruned = prune(q_node, total_pruned, config)

    # config.pruningCost to include the cost of the current node
    new_cost = first_step_reward + best_child_value - config.pruning_constant
    if new_cost > cost
      cost = new_cost
      pruned_action = q_node.action
    end
  end
  if pruned_action == -1
    total_pruned +=1
  end
  return cost, total_pruned
end

