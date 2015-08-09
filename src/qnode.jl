
# This type represents an AND-node (Q-node) of the belief tree, branching
# on observations. It maps the set of observations seen during simulations to
# the VNodes that the simulations reach at the next level

  # Fields:
  # obsToParticles: a mapping from observations seen to the particles (after
  # the transition) that generated the observations. The particles for each
  # observation become the representative set for the corresponding v-node
  # at the next level.
  # depth: depth of the v-node *above* this node.
  # action: action that led to this q-node.
  # firstStepReward: The average first step reward of particles when they
  # took action 'action'.
  # history: history up to the v-node *above* this node.
  # debug: Flag controlling debugging output.


type QNode
  obsToParticles::Dict{Int64, Array{DESPOTParticle,1}}
  depth::Int64
  action::Int64
  firstStepReward::Float64
  history::History
  weightSum::Float64
  obsToNode::Dict
  nVisits::Int64                # Needed for large problems
  
      # default constructor
      function QNode( problem::DESPOTProblem,
                      obsToParticles::Dict{Int64, Array{DESPOTParticle,1}},
                      depth::Int64,
                      action::Int64,
                      firstStepReward::Float64,
                      history::History,
                      config::Config)
                      
         this = new(
            # supplied variables
            obsToParticles,             # obsToParticles
            depth,                      # depth
            action,                     # action
            firstStepReward,            # firstStepReward
            history,                    # history

            # internal variables
            0,                          # weightSum
            Dict{Int64,VNode}(),        # obsToNode
            0                           # nVisits
            )
                
            this.weightSum = 0.
            
            for (obs, particles) in this.obsToParticles
                obsWs = 0.
                for p in particles
                    obsWs += p.wt
                end
                this.weightSum += obsWs

                add(history, action, obs)
                l::Float64, action::Int64 = lowerBound(problem, history, particles, depth + 1, config)
                u::Float64 = upperBound(problem, particles)
                removeLast(history)
                this.obsToNode[obs] = VNode(particles, l, u, this.depth + 1, obsWs, false, config) # TODO: check depth
            end
            return this
        end
end

function getUpperBound(qnode::QNode)
  ub = 0.
  for (obs,node) in qnode.obsToNode
      ub += node.ub * node.weight
  end
  return ub/qnode.weightSum
end

function getLowerBound(qnode::QNode)
  lb = 0.
  for (obs,node) in qnode.obsToNode
      lb += node.lb * node.weight
  end
  return lb/qnode.weightSum
end

#TODO: Fix this
function prune(qnode::QNode, totalPruned::Int64, config::Config)
  cost = 0.
  for (obs,node) in qnode.obsToNode
    cost, totalPruned += prune(node, totalPruned, config)
  end
  return cost, totalPruned
end
