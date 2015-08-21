
# This class ("Value Node") encapsulates a belief node (and recursively, a
# belief tree). It stores the set of particles associated with the node, an
# AND-node for each action, and some bookkeeping information.

type VNode
  particles::Array{DESPOTParticle,1}
  lb::Float64
  ub::Float64
  depth::Int64
  defaultValue::Float64             # Value of the default policy (= lbound value
                                    # before any backups are performed)
  prunedAction::Int64               # Best action at the node after pruning
  weight::Float64                   # Sum of particle weights at this belief
  bestUBAction::Int64               # Action that gives the highest upper bound
  inTree::Bool                      # True if the node is visited by Solver::trial().
                                    # In order to determine if a node is a fringe node
                                    # of the belief tree, we need to expand it one level.
                                    # The nodes added during this expansion of a fringe
                                    # node are not considered to be within the tree unless
                                    # explicitly visited by Solver::Trial(), so we use
                                    # this indicator variable.
  nTreeNodes::Int64                 # Number of nodes with inTree == true in the subtree
                                    # rooted at this node
  qnodes::Dict{Int,QNode}           # Dict of children q-nodes
  nVisits::Int64                    # Needed for large domains
  nActionsAllowed::Int64            # current number of action branches allowed in the node, needed for large domains
  qStar::Float64                    # best current Q-value, needed for large domains

  # default constructor
  function VNode( 
               particles::Array{DESPOTParticle,1},
               lBound::Float64,
               uBound::Float64,
               depth::Int64,
               weight::Float64,
               inTree::Bool,
               config::DESPOTConfig)

        this = new(
            particles,        # particles
            lBound,           # lBound
            uBound,           # uBound
            depth,            # depth
            lBound,           # defaultValue
            -1,               # pruned action
            weight,           # weight
            -1,               # bestUBAction
            inTree,           # inTree
            inTree ? 1:0,     # nTreeNodes
            Dict{Int,QNode}(),# qnodes
            0,                # nVisits
            0,                # nActionsAllowed
            -Inf,             # qStar
            )
        
        validateBounds(lBound, uBound, config)
        return this
  end
end

function getLowerBoundAction(node::VNode, config::DESPOTConfig)
  aStar = -1
  qStar = -Inf
  for (a,qnode) in node.qnodes
    remainingReward = getLowerBound(qnode)
    if qnode.firstStepReward + config.discount * remainingReward > qStar + config.tiny
      qStar = qnode.firstStepReward + config.discount * remainingReward
      aStar = a
    end
  end
  return aStar
end

#TODO: fix pruning

function prune(node::VNode, totalPruned::Int64, config::DESPOTConfig)
  # Cost if the node were pruned
  cost = (config.discount^node.depth) * node.weight * node.defaultValue
                - config.pruningConstant

  if !node.inTree # Leaf
    @assert(nodes.nTreeNodes == 0)
    return cost, totalPruned
  end

  for qnode in node.qnodes
    firstStepReward = config.discount^depth * weight * qnode.firstStepReward
    bestChildValue, totalPruned = prune(qnode, totalPruned, config)

    # config.pruningCost to include the cost of the current node
    newCost = firstStepReward + bestChildValue - config.pruningConstant
    if newCost > cost
      cost = newCost
      prunedAction = qnode.action
    end
  end
  if prunedAction == -1
    totalPruned +=1
  end
  return cost, totalPruned
end

