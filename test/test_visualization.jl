using POMDPs
using DESPOT
using POMDPToolbox
using POMDPModels

import DESPOT: bounds

immutable BabyBounds end
bounds{S}(::BabyBounds, p::BabyPOMDP, ::Vector{DESPOTParticle{S}}, ::DESPOTConfig) = (p.r_feed+p.r_hungry)/(1.0-discount(p)), 0

solver = DESPOTSolver{Bool, Bool, Bool, BabyBounds}(bounds = BabyBounds(),
                                                    random_streams=MersenneStreamArray(MersenneTwister(1)),
                                                    next_state=false,
                                                    curr_obs=false,
                                                    rng=Base.GLOBAL_RNG,
                                                    max_trials=100
                                                    )
problem = BabyPOMDP()

# test_solver is from POMDPToolbox
test_solver(solver, problem, updater=updater(problem))

blink(solver)
