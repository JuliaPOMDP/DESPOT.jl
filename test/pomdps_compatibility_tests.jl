using POMDPs
using DESPOT
using POMDPToolbox
using POMDPModels

import DESPOT: bounds

immutable BabyBounds end
bounds{S}(::BabyBounds, p::BabyPOMDP, ::Vector{DESPOTParticle{S}}, ::DESPOTConfig) = (p.r_feed+p.r_hungry)/(1.0-discount(p)), 0

solver = DESPOTSolver{Bool, Bool, Bool, BabyBounds}(bounds = BabyBounds(),
                                                    random_streams=MersenneStreamArray(MersenneTwister(1)),
                                                    root_default_action=false,
                                                    next_state=false,
                                                    curr_obs=false
                                                    )
problem = BabyPOMDP()

# test_solver is from POMDPToolbox
test_solver(solver, problem, updater=updater(problem))

test_solver(solver, problem)


immutable LightDarkBounds end
upper_bound{S}(::LightDarkUB, p::LightDark1D, ::Vector{DESPOTParticle{S}}, ::DESPOTConfig) = p.correct_r/(1.0-discount(p))

immutable LightDarkLB end
lower_bound{S}(::LightDarkLB, p::LightDark1D, ::Vector{DESPOTParticle{S}}, ::DESPOTConfig) = p.incorrect_r/(1.0-discount(p))

solver = DESPOTSolver{LightDark1DState, Int64, Float64, LightDarkLB, LightDarkUB}(ub=LightDarkUB(),
                                                                  lb=LightDarkLB(),
                                                                  random_streams=MersenneStreamArray(MersenneTwister(1)),
                                                                  root_default_action=false,
                                                                  next_state=LightDark1DState(),
                                                                  curr_obs=0.0
                                                                 )

test_solver(solver, LightDark1D())
