type DESPOTSolver <: Solver
  initialBelief::Array{StateProbability,1}
  bu::BeliefUpdate
  randomStreams::RandomStreams
  #history::History
  root::VNode
  rootDefaultAction::Int64
  nodeCount::Int64

  # default constructor
  function DESPOTSolver (initialBelief::Array{StateProbability,1},
                   bu::BeliefUpdate,
                   randomStreams::RandomStreams)
    this = new()
    # supplied variables
    this.initialBelief = initialBelief    # initialBelief
    this.bu = bu
    this.randomStreams = randomStreams    # randomStreams
    
    # internal variables
    # this.history = History()        # history
    # skip root initialization for now
    rootDefaultAction = -1 # rootDefaultAction
    return this
   end
end

function initSolver(solver::DESPOTSolver, problem::Problem, config::Config)

  # Construct particle pool
  belief = initialBelief(problem)
  particlePool = Array(Particle, 0)

  for b in belief
    push!(particlePool, Particle(b.s, 0, b.p))
  end

  shuffle!(particlePool)
  particles = sampleParticles(solver.bu, particlePool, config.nParticles, config)
  #newRoot(solver, problem, particles, config)
  return nothing
end

function newRoot(solver::DESPOTSolver, problem::Problem, particles::Array{Particle,1}, config::Config)
  
  lb::Float64, solver.rootDefaultAction = lowerBound(problem, solver.history, particles, 0, config)
  ub::Float64 = upperBound(problem, particles) #TODO: may need to put randomStreams back there
  solver.root = VNode(particles, lb, ub, 0, 1., false, config)

  return nothing
end


function search(solver::DESPOTSolver, problem::Problem, config::Config)
  nTrials = 0
  startTime = time()
  stopNow = false
 
  @printf("Before: lBound = %.10f, uBound = %.10f\n", solver.root.lb, solver.root.ub)
  while ((excessUncertainty(solver.root.lb,
                            solver.root.ub,
                            solver.root.lb,
                            solver.root.ub, 0, config) > 1e-6)
                            && !stopNow)

    #println("trial #$(nTrials)")
    trial(solver, problem, solver.root, nTrials, config)
    nTrials += 1
    
    if ((config.maxTrials > 0) && (nTrials >= config.maxTrials)) ||
       ((config.timePerMove > 0) && ((time() - startTime) >= config.timePerMove))
       
       stopNow = true
    end
  end

  @printf("After:  lBound = %.10f, uBound = %.10f\n", solver.root.lb, solver.root.ub)
  @printf("Number of trials: %d\n", nTrials)

  if (config.pruningConstant != 0)
    # Number of non-child belief nodes pruned
    totalPruned = prune(solver.root)
    act = solver.root.prunedAction
    return act == -1 ? solver.rootDefaultAction : act, currentTrials
  elseif !solver.root.inTree
      println("Root not in tree")
    return solver.rootDefaultAction, nTrials
  else
    return getLowerBoundAction(solver.root, config), nTrials
  end
end

function trial(solver::DESPOTSolver, problem::Problem, node::VNode, nTrials::Int64, config::Config)
    if (node.depth >= config.searchDepth) || isTerminal(problem, node.particles[1].state)
      return 0 # nodes added
    end
      
    if isempty(node.qnodes)
        expandOneStep(solver, problem, node, config)
    end

    aStar = node.bestUBAction
    nNodesAdded = 0
    oStar, weightedEuStar = getBestWEUO(node.qnodes[aStar], solver.root, config) # it's an array!
    
    if weightedEuStar > 0.
      add(solver.history, aStar, oStar)
      nNodesAdded = trial(solver, problem, node.qnodes[aStar].obsToNode[oStar], nTrials, config) # obsToNode is a Dict
      removeLast(solver.history)
    end
    node.nTreeNodes += nNodesAdded

    # Backup
    potentialLBound = node.qnodes[aStar].firstStepReward + config.discount * getLowerBound(node.qnodes[aStar])
    node.lb = max(node.lb, potentialLBound)

    # As the upper bound of a_star may become smaller than the upper bound of
    # another action, we need to check all actions - unlike the lower bound.
    node.ub = -Inf

    for a in 0:problem.nActions-1
        ub = node.qnodes[a].firstStepReward + config.discount * getUpperBound(node.qnodes[a]) # it's an array!
        if ub > node.ub
            node.ub = ub
            node.bestUBAction = a
        end
    end

    # Sanity check
    if (node.lb > node.ub + config.tiny)
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

function expandOneStep (solver::DESPOTSolver, problem::Problem, node::VNode, config::Config)
  
  qStar::Float64 = -Inf
  nextState::Int64 = -1
  reward::Float64 = 0.
  obs::Int64 = -1
  firstStepReward::Float64 = 0.
    
  for action in 0:problem.nActions-1

    obsToParticles = Dict{Int64,Vector{Particle}}()
    for p in node.particles
      nextState, reward, obs = step(problem,
                                    p.state,
                                    solver.randomStreams.streams[p.id+1,node.depth+1],
                                    action)

      if isTerminal(problem, nextState) && (obs != problem.kTerminal)
        error("Terminal state in a particle mismatches observation")
      end
      
      if !haskey(obsToParticles,obs)
          obsToParticles[obs] = Particle[]
      end
      tempParticle = Particle(nextState,p.id,p.wt)
      push!(obsToParticles[obs], tempParticle)
      firstStepReward += reward * p.wt
    end
    
    firstStepReward /= node.weight
    newQNode = QNode(problem, obsToParticles, node.depth, action, firstStepReward, solver.history, config)
    node.qnodes[action] = newQNode

    remainingReward = getUpperBound(newQNode)
    if (firstStepReward + config.discount*remainingReward) > (qStar + config.tiny)
      qStar = firstStepReward + config.discount * remainingReward
      node.bestUBAction = action
    end
    
    firstStepReward = 0.
  end
  return node
end

function updateBelief (solver::DESPOTSolver, problem::Problem, action::Int64, obs::Int64, config::Config)
  particles = beliefUpdateParticle(problem,
                                   solver.bu,
                                   solver.root.particles,
                                   action,
                                   obs,
                                   config)
                                   
  add(solver.history, action, obs)
  newRoot(solver, problem, particles, config)
end

function finished(solver::DESPOTSolver, problem::Problem)
  for p in solver.root.particles
    if !isTerminal(problem, p.state)
      return false
    end
  end
  return true
end
