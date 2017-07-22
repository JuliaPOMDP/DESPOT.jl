using POMDPs
using DESPOT
using POMDPToolbox
using POMDPModels

import DESPOT: bounds

struct BabyBounds end
bounds{S}(::BabyBounds, p::BabyPOMDP, ::Vector{DESPOTParticle{S}}, ::DESPOTConfig) = (p.r_feed+p.r_hungry)/(1.0-discount(p)), 0

solver = DESPOTSolver{Bool, Bool, Bool, BabyBounds, MersenneStreamArray}(bounds = BabyBounds(),
                                                    random_streams=MersenneStreamArray(MersenneTwister(1)),
                                                    next_state=false,
                                                    curr_obs=false,
                                                    rng=Base.GLOBAL_RNG
                                                    )
problem = BabyPOMDP()

# test_solver is from POMDPToolbox
test_solver(solver, problem, updater=updater(problem))

test_solver(solver, problem)


struct LightDarkBounds end
bounds{S}(::LightDarkBounds, p::LightDark1D, ::Vector{DESPOTParticle{S}}, ::DESPOTConfig) = p.incorrect_r/(1.0-discount(p)), p.correct_r/(1.0-discount(p))

solver = DESPOTSolver{LightDark1DState, Int64, Float64, LightDarkBounds, MersenneStreamArray}(bounds=LightDarkBounds(),
                                                                  random_streams=MersenneStreamArray(MersenneTwister(1)),
                                                                  next_state=LightDark1DState(0, 0.0),
                                                                  curr_obs=0.0
                                                                 )

test_solver(solver, LightDark1D())
