
# import DESPOT:
#     upper_bound,
#     init_upper_bound

type UpperBoundNonStochastic <: DESPOTUpperBound #TODO: perhaps make parametric 

    upper_bound_act::Vector
    upper_bound_memo::Vector{Float64}
    
    # Constructor
    function UpperBoundNonStochastic(pomdp::POMDP)
    
        this = new()
        
        # this executes just once per problem run
        this.upper_bound_act = Array(typeof(create_action(pomdp)), n_states(pomdp))    # upper_bound_act
#        fill!(this.upper_bound_act, 0)
        this.upper_bound_memo = Array(Float64, n_states(pomdp)) # upper_bound_memo

    return this
  end
end

function DESPOT.init_upper_bound(ub::UpperBoundNonStochastic,
                    pomdp::POMDP,
                    config::DESPOTConfig)
    
    current_level_ub_memo = Array(Float64, n_states(pomdp))
    next_level_ub_memo = Array(Float64, n_states(pomdp))
    
    next_state = create_state(pomdp)
    r::Float64 = 0.0
    trans_distribution = create_transition_distribution(pomdp)
    rng = DESPOTRandomNumber(0) # dummy RNG
    
    fill!(current_level_ub_memo, -Inf)
    
    for s in iterator(states(pomdp))
        next_level_ub_memo[index(pomdp,s)+1] = fringe_upper_bound(pomdp,s) # 1-based indexing
    end
    

    for i in 1:config.search_depth # length of horizon
        for s in iterator(states(pomdp))
            for a in iterator(actions(pomdp))
                trans_distribution.state = s #TODO: this might not be necessary - do by reference
                trans_distribution.action = a #TODO: this might not be necessary - do by reference
  #              println("s: $(trans_distribution.state.index), a: $(trans_distribution.action.index)")
                rand!(rng, next_state, trans_distribution)
                r = reward(pomdp, s, a)
                possibly_improved_value = r + pomdp.discount * next_level_ub_memo[next_state.index+1]
                if (possibly_improved_value > current_level_ub_memo[s.index+1])
                    current_level_ub_memo[s.index+1] = possibly_improved_value
                    if i == config.search_depth
                        # Set best actions when last level is being computed
                        ub.upper_bound_act[s.index+1] = deepcopy(a) #TODO: consider replacing with a custom copy for speed
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

function DESPOT.upper_bound(ub::UpperBoundNonStochastic,
                     pomdp::POMDP,
                     #particles::Vector{Particle}, #TODO: figure out why this does not work
                     particles::Vector,
                     config::DESPOTConfig)
  weight_sum = 0.
  total_cost = 0.

  for p in particles
    weight_sum += p.weight
    total_cost += p.weight * ub.upper_bound_memo[p.state.index+1]
  end
  return total_cost / weight_sum
end
