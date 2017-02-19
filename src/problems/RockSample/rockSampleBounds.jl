# Upper-level bounds type for the RockSample problem

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
    lbound::Float64, best_action::RockSampleAction =
        lower_bound_and_action(b.lb, pomdp, particles, b.ub.upper_bound_act, config)
       
    return lbound, ubound, best_action
end

function init_bounds(bounds::RockSampleBounds, pomdp::RockSample, config::DESPOTConfig)
    init_bound(bounds.ub, pomdp, config)
    # lower bound init not currently needed
end
