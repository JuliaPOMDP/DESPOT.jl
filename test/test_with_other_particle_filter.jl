using ParticleFilters

pomdp = RockSample(4, 4)

filter = SIRParticleFilter(pomdp, 500)

custom_lb = RockSampleParticleLB{RockSampleState, RockSampleAction, RockSampleObs}(pomdp) # custom lower bound to use with DESPOT solver
custom_ub = UpperBoundNonStochastic{RockSampleState, RockSampleAction, RockSampleObs}(pomdp)
  
solver = DESPOTSolver{RockSampleState,
                      RockSampleAction,
                      RockSampleObs,
                      RockSampleParticleLB,
                      UpperBoundNonStochastic}(lb = custom_lb, ub = custom_ub)

policy = solve(solver, pomdp)

ro = RolloutSimulator(max_steps=10, rng=MersenneTwister(1))

simulate(ro, pomdp, policy, filter)
