using POMDPs
using DESPOT
using POMDPToolbox
using POMDPModels
using ParticleFilters

import DESPOT: upper_bound, lower_bound

# Note: init_lower_bound and init_upper_bound should have a default implementation that does nothing
immutable BabyUB <: DESPOTUpperBound{Bool, Bool, Bool} end
upper_bound{S}(::BabyUB, p::BabyPOMDP, ::Vector{DESPOTParticle{S}}, ::DESPOTConfig) = 0

immutable BabyLB <: DESPOTLowerBound{Bool, Bool, Bool} end
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
# test_solver(solver, problem)


immutable LightDarkUB <: DESPOTUpperBound{LightDark1DState, Int64, Float64} end
upper_bound{S}(::LightDarkUB, p::LightDark1D, ::Vector{DESPOTParticle{S}}, ::DESPOTConfig) = p.correct_r/(1.0-discount(p))

immutable LightDarkLB <: DESPOTLowerBound{LightDark1DState, Int64, Float64} end
lower_bound{S}(::LightDarkLB, p::LightDark1D, ::Vector{DESPOTParticle{S}}, ::DESPOTConfig) = p.incorrect_r/(1.0-discount(p))

solver = DESPOTSolver{LightDark1DState, Int64, Float64, LightDarkLB, LightDarkUB}(ub=LightDarkUB(),
                                                                  lb=LightDarkLB(),
                                                                  random_streams=MersenneStreamArray(MersenneTwister(1)),
                                                                  root_default_action=false,
                                                                  next_state=LightDark1DState(),
                                                                  curr_obs=0.0
                                                                 )

test_solver(solver, LightDark1D(), updater=SIRParticleFilter(LightDark1D(), 100))

# Usability notes:
# It would be much easier to use the DESPOTSolver if the user did not have to
# specify the type parameters up front, and designate the root_default_action,
# next_state, and curr_obs, and I think it is technically feasible to remove
# these requirements, but ease-of-use is a secondary goal to getting the solver
# to work with a variety of problems.
