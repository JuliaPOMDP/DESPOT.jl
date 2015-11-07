import DESPOT:
    upper_bound,
    init_upper_bound

type UpperBoundNonStochastic <: DESPOTUpperBound

    upper_bound_act::Vector{Int64}
    upper_bound_memo::Vector{Float64}
    
    # Constructor
    function UpperBoundNonStochastic(pomdp::POMDP)
    
        this = new()
        
        # this executes just once per problem run
        this.upper_bound_act = Array(Int64, n_states(pomdp))    # upper_bound_act
        fill!(this.upper_bound_act, 0)
        this.upper_bound_memo = Array(Float64, n_states(pomdp)) # upper_bound_memo

    return this
  end
end

function init_upper_bound(ub::UpperBoundNonStochastic,
                    pomdp::POMDP,
                    config::DESPOTConfig)
                          
    current_level_ub_memo = Array(Float64, n_states(pomdp))
    next_state = create_state(pomdp)
    r::Float64 = 0.0
    trans_distribution = create_transition_distribution(pomdp)
    rng = DESPOTRandomNumber(0) # dummy RNG
    
    fill!(current_level_ub_memo, -Inf)

    next_level_ub_memo = [fringe_upper_bound(pomdp, s) for s = 0:n_states(pomdp)-1]
    
    for i in 1:config.search_depth # length of horizon
        for s in states(pomdp)
            for a in actions(pomdp)
                trans_distribution.state = s
                trans_distribution.action = a
                next_state = rand!(rng, next_state, trans_distribution)
                r = reward(pomdp, s, a)
                possibly_improved_value = r + pomdp.discount * next_level_ub_memo[next_state+1]
                if (possibly_improved_value > current_level_ub_memo[s+1])
                    current_level_ub_memo[s+1] = possibly_improved_value
                    if i == config.search_depth
                        # Set best actions when last level is being computed
                        ub.upper_bound_act[s+1] = a
                    end
                end
            end # for a
        end #for s
        
        # swap array references
        tmp = current_level_ub_memo
        current_level_ub_memo = next_level_ub_memo
        next_level_ub_memo = tmp
        
        fill!(current_level_ub_memo,-Inf)
    end

    #TODO: this can probably be done more optimally (by referencing rs.upper_bound_memo to start with),
    # however, this only runs once per problem and is probably not a big deal. Leave it as is for now.
    copy!(ub.upper_bound_memo, next_level_ub_memo)
    
    return nothing
end

function upper_bound(ub::UpperBoundNonStochastic,
                     pomdp::POMDP,
                     #particles::Vector{Particle}, #TODO: figure out why this does not work
                     particles::Vector,
                     config::DESPOTConfig)
  weight_sum = 0.
  total_cost = 0.

  for p in particles
    weight_sum += p.weight
    total_cost += p.weight * ub.upper_bound_memo[p.state+1]
  end
  return total_cost / weight_sum
end
