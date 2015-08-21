using DESPOT
import DESPOT: update_belief!

type DESPOTSolver <: Solver
    initialBelief::DESPOTBelief
    bu::DESPOTBeliefUpdate
    randomStreams::RandomStreams
    #history::History
    root::VNode
    rootDefaultAction::Int64
    nodeCount::Int64

  # default constructor
    function DESPOTSolver (pomdp::DESPOTPomdp)

        this = new()
        # supplied variables
        this.initialBelief = initial_belief(pomdp.problem)
        this.randomStreams = pomdp.randomStreams
        
        # internal variables
        # this.history = History()        # history
        # skip root initialization for now
        rootDefaultAction = -1 # rootDefaultAction
        return this
    end
end

function initSolver(solver::DESPOTSolver, pomdp::DESPOTPomdp)
#function initSolver(policy::DESPOTPolicy)

  # Construct particle pool
  belief = initial_belief(pomdp.problem)
  particlePool = belief.particles

#   #TODO: cleanup particles vs state probabilities stuff - it can be simpler
#   for b in belief.particles
#     push!(particlePool, DESPOTParticle(b.s, 0, b.p))
#   end

  shuffle!(particlePool)
  particles = sampleParticles(pomdp.bu, particlePool, pomdp.config.nParticles, pomdp.config)
  
  newRoot(solver, pomdp, particles)
  return nothing
end

function newRoot(solver::DESPOTSolver, pomdp::DESPOTPomdp, particles::Array{DESPOTParticle,1})
  
  lb::Float64, solver.rootDefaultAction = lower_bound(pomdp.problem, particles, 0, pomdp.config)
  ub::Float64 = upper_bound(pomdp, particles)
  solver.root = VNode(particles, lb, ub, 0, 1., false, pomdp.config)

  return nothing
end


function search(solver::DESPOTSolver, pomdp::DESPOTPomdp)
  nTrials = 0
  startTime = time()
  stopNow = false
 
  @printf("Before: lBound = %.10f, uBound = %.10f\n", solver.root.lb, solver.root.ub)
  while ((excessUncertainty(solver.root.lb,
                            solver.root.ub,
                            solver.root.lb,
                            solver.root.ub, 0, pomdp.config) > 1e-6)
                            && !stopNow)

    #println("trial #$(nTrials)")
    trial(solver, pomdp, solver.root, nTrials)
    nTrials += 1
    
    if ((pomdp.config.maxTrials > 0) && (nTrials >= pomdp.config.maxTrials)) ||
       ((pomdp.config.timePerMove > 0) && ((time() - startTime) >= pomdp.config.timePerMove))
       
       stopNow = true
    end
  end

  @printf("After:  lBound = %.10f, uBound = %.10f\n", solver.root.lb, solver.root.ub)
  @printf("Number of trials: %d\n", nTrials)

  if (pomdp.config.pruningConstant != 0)
    # Number of non-child belief nodes pruned
    totalPruned = prune(solver.root)
    act = solver.root.prunedAction
    return act == -1 ? solver.rootDefaultAction : act, currentTrials
  elseif !solver.root.inTree
      println("Root not in tree")
    return solver.rootDefaultAction, nTrials
  else
    return getLowerBoundAction(solver.root, pomdp.config), nTrials
  end
end

function trial(solver::DESPOTSolver, pomdp::DESPOTPomdp, node::VNode, nTrials::Int64)
    if (node.depth >= pomdp.config.searchDepth) || isterminal(pomdp.problem, node.particles[1].state)
      return 0 # nodes added
    end
      
    if isempty(node.qnodes)
        expandOneStep(solver, pomdp, node)
    end

    aStar = node.bestUBAction
    nNodesAdded = 0
    oStar, weightedEuStar = getBestWEUO(node.qnodes[aStar], solver.root, pomdp.config) # it's an array!
    
    if weightedEuStar > 0.
      add(pomdp.history, aStar, oStar)
      nNodesAdded = trial(solver,
                          pomdp,
                          node.qnodes[aStar].obsToNode[oStar], # obsToNode is a Dict
                          nTrials) 
      removeLast(pomdp.history)
    end
    node.nTreeNodes += nNodesAdded

    # Backup
    potentialLBound = node.qnodes[aStar].firstStepReward + pomdp.config.discount * getLowerBound(node.qnodes[aStar])
    node.lb = max(node.lb, potentialLBound)

    # As the upper bound of a_star may become smaller than the upper bound of
    # another action, we need to check all actions - unlike the lower bound.
    node.ub = -Inf

    for a in 0:pomdp.problem.nActions-1
        ub = node.qnodes[a].firstStepReward + pomdp.config.discount * getUpperBound(node.qnodes[a]) # it's an array!
        if ub > node.ub
            node.ub = ub
            node.bestUBAction = a
        end
    end

    # Sanity check
    if (node.lb > node.ub + pomdp.config.tiny)
      println ("depth = $(node.depth)")
      #error("Lower bound ($(node.lb)) is higher than upper bound ($(node.ub))")
      warn("Lower bound ($(node.lb)) is higher than upper bound ($(node.ub))")
    end

    if !node.inTree
      node.inTree = true
      node.nTreeNodes += 1
      nNodesAdded +=1
    end
    return nNodesAdded
end

function expandOneStep (solver::DESPOTSolver, pomdp::DESPOTPomdp, node::VNode)
  
  qStar::Float64 = -Inf
  nextState::Int64 = -1
  reward::Float64 = 0.
  obs::Int64 = -1
  firstStepReward::Float64 = 0.
    
  for action in 0:pomdp.problem.nActions-1

    obsToParticles = Dict{Int64,Vector{DESPOTParticle}}()
    for p in node.particles
      nextState, reward, obs = step(pomdp.problem,
                                    p.state,
                                    solver.randomStreams.streams[p.id+1,node.depth+1],
                                    action)

      if isterminal(pomdp.problem, nextState) && (obs != pomdp.problem.kTerminal)
        error("Terminal state in a particle mismatches observation")
      end
      
      if !haskey(obsToParticles,obs)
          obsToParticles[obs] = DESPOTParticle[]
      end
      tempParticle = DESPOTParticle(nextState,p.id,p.wt)
      push!(obsToParticles[obs], tempParticle)
      firstStepReward += reward * p.wt
    end
    
    firstStepReward /= node.weight
    #newQNode = QNode(pomdp.problem, obsToParticles, node.depth, action, firstStepReward, pomdp.history, pomdp.config)
    newQNode = QNode(pomdp, obsToParticles, node.depth, action, firstStepReward)
    node.qnodes[action] = newQNode

    remainingReward = getUpperBound(newQNode)
    if (firstStepReward + pomdp.config.discount*remainingReward) > (qStar + pomdp.config.tiny)
      qStar = firstStepReward + pomdp.config.discount * remainingReward
      node.bestUBAction = action
    end
    
    firstStepReward = 0.
  end
  return node
end


