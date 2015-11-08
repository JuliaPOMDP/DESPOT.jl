import DESPOT:
        lower_bound

type DESPOTDefaultLowerBound <: DESPOTLowerBound
    #placeholder for now
end

function init_bound(ub::DESPOTDefaultLowerBound,
                    pomdp::POMDP,
                    config::DESPOTConfig)
    error("Function init_bound for $(typeof(ub)) has not been implemented yet")                    
end

function lower_bound(lb::DESPOTDefaultLowerBound,
                     pomdp::POMDP,
                     particles::Vector{DESPOTParticle},
                     config::DESPOTConfig)
    error("Function lower_bound for $(typeof(lb)) has not been implemented yet")
end
