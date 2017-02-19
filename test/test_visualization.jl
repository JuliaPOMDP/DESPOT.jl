using POMDPs
using DESPOT
using POMDPToolbox
using POMDPModels

import DESPOT: upper_bound, lower_bound

# Note: init_lower_bound and init_upper_bound should have a default implementation that does nothing
immutable BabyUB end
upper_bound{S}(::BabyUB, p::BabyPOMDP, ::Vector{DESPOTParticle{S}}, ::DESPOTConfig) = 0

immutable BabyLB end
lower_bound{S}(::BabyLB, p::BabyPOMDP, ::Vector{DESPOTParticle{S}}, ::DESPOTConfig) = (p.r_feed+p.r_hungry)/(1.0-discount(p))

solver = DESPOTSolver{Bool, Bool, Bool, BabyLB, BabyUB}(ub = BabyUB(),
                                                        lb = BabyLB(),
                                                        random_streams=MersenneStreamArray(MersenneTwister(1)),
                                                        root_default_action=false,
                                                        next_state=false,
                                                        curr_obs=false
                                                       )
problem = BabyPOMDP()

# test_solver is from POMDPToolbox
test_solver(solver, problem, updater=updater(problem))

blink(solver)
