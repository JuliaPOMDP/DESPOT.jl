type DESPOTConfig
  # Maximum depth of the search tree
  search_depth::Int64
  # Random-number seed
  main_seed::Uint32
  # Amount of CPU time used for search during each move. Does not include the
  # time taken to prune the tree and update the belief.
  time_per_move::Float64
  # Number of starting states (samples)
  n_particles::Int64
  # Regularization parameter
  pruning_constant::Float64
  # Parameter such that eta * width(root) is the target uncertainty at the
  # root of the search tree, used in determining when to terminate a trial.
  eta::Float64
  # Number of moves to simulate
  sim_len::Int64
  # Whether the initial upper bound is approximate or true. If approximate,
  # the solver allows initial lower bound > initial upper bound at a node.
  approximate_ubound::Bool
#   particle_weight_threshold::Float64
#   eff_particle_fraction::Float64
  tiny::Float64 # tiny number
  max_trials::Int64 
  rand_max::Int64 
  debug::Uint8
  
  # construct empty
  function DESPOTConfig()
   this = new()
   return this
  end
end
