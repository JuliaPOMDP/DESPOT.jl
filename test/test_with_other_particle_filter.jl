using ParticleFilters

pomdp = RockSample(4, 4)

filter = SIRParticleFilter(pomdp, 500)

custom_bounds = RockSampleBounds(pomdp) # custom lower bound to use with DESPOT solver
  
solver = DESPOTSolver{RockSampleState,
                      RockSampleAction,
                      RockSampleObs,
                      RockSampleBounds}(bounds = custom_bounds)

policy = solve(solver, pomdp)

ro = RolloutSimulator(max_steps=10, rng=MersenneTwister(1))

simulate(ro, pomdp, policy, filter)
