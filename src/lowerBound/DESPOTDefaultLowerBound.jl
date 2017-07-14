import DESPOT:
        lower_bound

mutable struct DESPOTDefaultLowerBound
    #placeholder for now
end

function init_bound{S,A,O}(ub::DESPOTDefaultLowerBound,
                    pomdp::POMDP{S,A,O},
                    config::DESPOTConfig)
    error("Function init_bound for $(typeof(ub)) has not been implemented yet")
end

function lower_bound{S,A,O}(lb::DESPOTDefaultLowerBound,
                     pomdp::POMDP{S,A,O},
                     particles::Vector{DESPOTParticle{S}},
                     config::DESPOTConfig)
    error("Function lower_bound for $(typeof(lb)) has not been implemented yet")
end
