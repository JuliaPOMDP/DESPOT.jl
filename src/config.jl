type DESPOTConfig
  # Maximum depth of the search tree
  searchDepth::Uint32
  # Discount factor
  discount::Float64
  # Random-number seed
  rootSeed::Uint32
  # Amount of CPU time used for search during each move. Does not include the
  # time taken to prune the tree and update the belief.
  timePerMove::Float64
  # Number of starting states (samples)
  nParticles::Uint32
  # Regularization parameter
  pruningConstant::Float64
  # Parameter such that eta * width(root) is the target uncertainty at the
  # root of the search tree, used in determining when to terminate a trial.
  eta::Float64
  # Number of moves to simulate
  simLen::Int64
  # Whether the initial upper bound is approximate or true. If approximate,
  # the solver allows initial lower bound > initial upper bound at a node.
  approximateUBound::Bool
  particleWtThreshold::Float64
  numEffParticleFraction::Float64
  tiny::Float64 # tiny number
  maxTrials::Int64 
  randMax::Int64 
  debug::Uint8
  
  # construct empty
  function DESPOTConfig()
   this = new()
   return this
  end
end
