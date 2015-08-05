
function excessUncertainty(l::Float64,
                           u::Float64,
                           rootL::Float64,
                           rootU::Float64,
                           depth::Int64,
                           config::Config)

  eu =  (u-l) - #width of current node
        (config.eta * (rootU-rootL)) * # epsilon
        (config.discount^(-depth))
  return eu
end

# Returns the observation with the highest weighted excess uncertainty
# ("WEU"), along with the value of the WEU.
# root: Root of the search tree, passed to facilitate computation of the
# excess uncertainty

function getBestWEUO(qnode::QNode, root::VNode, config::Config)
  weightedEuStar = -Inf
  oStar = 0.
  
  for (obs,node) in qnode.obsToNode
        weightedEu = node.weight / qnode.weightSum *
                            excessUncertainty(
                            node.lb, node.ub,
                            root.lb, root.ub,
                            qnode.depth+1, config)
        
        if weightedEu > weightedEuStar
            weightedEuStar = weightedEu
            oStar = obs
        end
  end
  return oStar, weightedEuStar
end

# Get WEUO for a single observation branch
function getNodeWEUO(qnode::QNode, root::VNode, obs::Int64)
   weightedEu = qnode.obsToNode[obs].weight / qnode.weightSum *
                        excessUncertainty(
                        qnode.obsToNode[obs].lb, qnode.obsToNode[obs].ub,
                        root.lb, root.ub, qnode.depth+1)
   return weightedEu
end

# Returns the v-node corresponding to a given observation
function belief(qnode::QNode, obs::Int64)
  return qnode.obsToNode[obs]
end

function validateBounds(lb::Float64, ub::Float64, config::Config)
  if (ub >= lb)
    return
  end

  if (ub > lb - config.tiny) || config.approximateUBound
    ub = lb
  else
    @assert(false)
  end
end

function almostTheSame(x::Float64,y::Float64,config::Config)
  return abs(x-y) < config.tiny
end
