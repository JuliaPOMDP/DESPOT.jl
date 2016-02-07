

type UpperBoundNonStochastic{S,A,O} <: DESPOTUpperBound{S,A,O}
    upper_bound_act::Vector{A}
    upper_bound_memo::Vector{Float64}
    
    # Constructor
    function UpperBoundNonStochastic(pomdp::POMDP{S,A,O})    
        this = new()       
        # this executes just once per problem run
        this.upper_bound_act = Array{A}(n_states(pomdp))    # upper_bound_act
        this.upper_bound_memo = Array{Float64}(n_states(pomdp)) # upper_bound_memo
        return this
    end
end

function DESPOT.init_upper_bound{S,A,O}(ub::UpperBoundNonStochastic{S,A,O},
                    pomdp::POMDP{S,A,O},
                    config::DESPOTConfig)
    
    current_level_ub_memo = Array(Float64, n_states(pomdp))
    next_level_ub_memo = Array(Float64, n_states(pomdp))
    
    next_state = S()
    r::Float64 = 0.0
    trans_distribution = create_transition_distribution(pomdp)
    rng = DESPOTRandomNumber(0) # dummy RNG
    
    fill!(current_level_ub_memo, -Inf)
    
    for s in iterator(states(pomdp))
        next_level_ub_memo[state_index(pomdp,s)+1] = fringe_upper_bound(pomdp,s) # 1-based indexing
    end
    
    for i in 1:config.search_depth # length of horizon
        for s in iterator(states(pomdp))
            for a in iterator(actions(pomdp))
                trans_distribution.state = s
                trans_distribution.action = a
                next_state = POMDPs.rand(rng, trans_distribution, next_state)
                r = reward(pomdp, s, a)
                possibly_improved_value = 
                    r + pomdp.discount * next_level_ub_memo[state_index(pomdp,next_state)+1]
                if (possibly_improved_value > current_level_ub_memo[state_index(pomdp,s)+1])
                    current_level_ub_memo[state_index(pomdp,s)+1] = possibly_improved_value
                    if i == config.search_depth
                        # Set best actions when last level is being computed
                        ub.upper_bound_act[state_index(pomdp,s)+1] = a
                    end
                end
            end # for a
        end # for s
        
        # swap array references
        tmp = current_level_ub_memo
        current_level_ub_memo = next_level_ub_memo
        next_level_ub_memo = tmp
        
        fill!(current_level_ub_memo,-Inf)
    end

    #TODO: this can probably be done more optimally (by referencing upper_bound_memo to start with),
    # however, this only runs once per problem and is probably not a big deal. Leave it as is for now.
    copy!(ub.upper_bound_memo, next_level_ub_memo)
    
    return nothing
end

function DESPOT.upper_bound{S,A,O}(ub::UpperBoundNonStochastic{S,A,O},
                     pomdp::POMDP{S,A,O},
                     particles::Vector{DESPOTParticle{S}},
                     config::DESPOTConfig)
  weight_sum = 0.
  total_cost = 0.

  for p in particles
    weight_sum += p.weight
    total_cost += p.weight * ub.upper_bound_memo[state_index(pomdp,p.state)+1]
  end
  return total_cost / weight_sum
end
