using ParticleFilters

pomdp = RockSample(4, 4)

filter = SIRParticleFilter(pomdp, 500)

custom_bounds = RockSampleBounds(pomdp) # custom bounds for use with DESPOT solver
  
solver = DESPOTSolver{RockSampleState,
                      RockSampleAction,
                      RockSampleObs,
                      RockSampleBounds,
                      MersenneStreamArray
                     }(bounds = custom_bounds,
                                        random_streams = MersenneStreamArray(MersenneTwister(2)) 
                                       )

policy = solve(solver, pomdp)

ro = RolloutSimulator(max_steps=10, rng=MersenneTwister(1))

simulate(ro, pomdp, policy, filter)
