# Combined (lower and upper) bounds type for the RockSample problem
include("rockSampleParticleLB.jl")
include("rockSampleFringeUB.jl")
include("../../upperBound/upperBoundNonStochastic.jl")

import DESPOT: bounds, init_bounds

type RockSampleBounds
    lb::RockSampleParticleLB
    ub::UpperBoundNonStochastic{RockSampleState,RockSampleAction,RockSampleObs}
    
    function RockSampleBounds(pomdp::RockSample)
        this = new()
        this.lb = RockSampleParticleLB(pomdp)
        this.ub = UpperBoundNonStochastic{RockSampleState,RockSampleAction,RockSampleObs}(pomdp)

        return this
    end
end

function bounds(b::RockSampleBounds,
       pomdp::RockSample,
       particles::Vector{DESPOTParticle{RockSampleState}},
       config::DESPOTConfig)
       
    ubound::Float64 = upper_bound(b.ub, pomdp, particles, config)
    lbound::Float64 = lower_bound(b.lb, pomdp, particles, b.ub.upper_bound_act, config)
       
    return lbound, ubound
end

function init_bounds(bounds::RockSampleBounds, pomdp::RockSample, config::DESPOTConfig)
    init_bound(bounds.ub, pomdp, config)
    # lower bound init not currently needed
end
