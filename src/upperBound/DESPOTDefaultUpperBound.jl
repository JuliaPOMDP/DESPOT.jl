type DESPOTDefaultUpperBound <: DESPOTUpperBound
    #placeholder for now
end

function init_bound(ub::UpperBoundNonStochastic,
                    pomdp::POMDP,
                    config::DESPOTConfig)
    error("Function init_bound for $(typeof(ub)) has not been implemented yet")                    
end

function lower_bound(lb::DESPOTDefaultUpperBound,
                     pomdp::POMDP,
                     particles::Vector{DESPOTParticle})
    error("Function lower_bound for $(typeof(lb)) has not been implemented yet")
end
