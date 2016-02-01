  # A RockSample lower bound estimation procedure for solvers that use
  # weighted particle belief representation (defined in types.jl)
  # TODO: think how to make this more generic
  #
  # Strategy: Compute a representative state by setting the state of
  # each rock to the one that occurs more frequently in the particle set.
  # Then compute the best sequence of actions for the resulting
  # state. Apply this sequence of actions to each particle and average
  # to get a lower bound.
  #
  # Possible improvement: If a rock is sampled while replaying the action
  # sequence, use dynamic programming to look forward in the action
  # sequence to determine if it would be a better idea to first sense the
  # rock instead. (sensing eliminates the bad rocks in the particle set)

import DESPOT:
        lower_bound,
        init_lower_bound
        
type RockSampleParticleLB{S,A,O} <: DESPOTLowerBound{S,A,O}
    weight_sum_of_state::Vector{Float64}
    
    function RockSampleParticleLB(pomdp::RockSample)
        this = new()
        this.weight_sum_of_state = Array(Float64, pomdp.n_states)
        fill!(this.weight_sum_of_state, -Inf)
        return this
    end
end

function init_lower_bound{S,A,O}(lb::RockSampleParticleLB{S,A,O},
                    pomdp::RockSample,
                    config::DESPOTConfig)
    # nothing to do for now
end

function lower_bound{S,A,O}(lb::RockSampleParticleLB{S,A,O},
                     pomdp::RockSample,
                     particles::Vector{DESPOTParticle{S}},
                     ub_actions::Vector{A},
                     config::DESPOTConfig)
    
    weight_sum::Float64 = 0.0
    state_seen::Dict{Int64,Int64} = Dict{Int64,Int64}()
    s_index::Int64 = -1
    
    # Since for this problem the cell that the rover is in is deterministic, picking pretty much
    # any particle state is ok
    if length(particles) > 0
        if isterminal(pomdp, particles[1].state)
            return 0.0, RockSampleAction(-1) # lower bound value and best action
        end
    end

    # The expected value of sampling a rock, over all particles
    expected_sampling_value::Array{Float64} = fill(0.0, pomdp.n_rocks)
    seen_ptr::Int64 = 0

    # Compute the expected sampling value of each rock. Instead of factoring
    # the weight of each particle, we first record the weight of each state.
    # This is so that the inner loop that updates the expected value of each
    # rock runs once per state seen, instead of once per particle seen. If
    # there are lots of common states between particles, this gives a
    # significant speedup to the search because the lower bound is the
    # bottleneck.

    for p in particles
        if lb.weight_sum_of_state[p.state.index+1] == -Inf #Array
            lb.weight_sum_of_state[p.state.index+1] = p.weight
            state_seen[seen_ptr] = p.state.index
            seen_ptr += 1
        else
            lb.weight_sum_of_state[p.state.index+1] += p.weight;
        end
    end
    
    for i in 0:(seen_ptr-1)
        s_index = state_seen[i]
        weight_sum += lb.weight_sum_of_state[s_index+1]
        for j in 0:pomdp.n_rocks-1
            expected_sampling_value[j+1] += lb.weight_sum_of_state[s_index+1] * (rock_status(j, s_index) ? 10.0 : -10.0)
        end
    end
    
    # Reset for next use
    fill!(lb.weight_sum_of_state, -Inf)

    most_likely_rock_set = 0
    for i in 0:pomdp.n_rocks-1
        expected_sampling_value[i+1] /= weight_sum
        # Threshold the average to good or bad
        if expected_sampling_value[i+1] > -config.tiny
            most_likely_rock_set |= (1 << i)
        end
        if abs(expected_sampling_value[i+1]) < config.tiny
            expected_sampling_value[i+1] = 0.0
        end
    end

    # Since for this problem the cell that the rover is in is deterministic, picking pretty much
    # any particle state is ok
    s = RockSampleState(
        make_state_index(pomdp, cell_of(pomdp, particles[1].state), most_likely_rock_set))

    # Sequence of actions taken in the optimal policy
    optimal_policy = Vector{RockSampleAction}()
    ret::Float64 = 0.0
    reward::Float64 = 0.0
    prev_cell_coord = [0,0] # initial value - should cause error if not properly assigned
#    next_s::RockSampleState = create_state(pomdp)
    next_s::RockSampleState = RockSampleState()
    r::Float64 = 0.0
    trans_distribution = create_transition_distribution(pomdp)
    rng = DESPOTRandomNumber(0) # dummy RNG
    
    while true
        a = ub_actions[s.index+1]
        trans_distribution.state = s
        trans_distribution.action = a
        next_s = POMDPs.rand(rng, next_s, trans_distribution)       
        if isterminal(pomdp, next_s)
            prev_cell_coord[1] = pomdp.cell_to_coords[cell_of(pomdp, s)+1][1]
            prev_cell_coord[2] = pomdp.cell_to_coords[cell_of(pomdp, s)+1][2]
            ret = 10.0
            break
        end
        push!(optimal_policy, a)
        if length(optimal_policy) == config.search_depth
            prev_cell_coord[1] = pomdp.cell_to_coords[cell_of(pomdp, next_s)+1][1]
            prev_cell_coord[2] = pomdp.cell_to_coords[cell_of(pomdp, next_s)+1][2]
            ret = 0.0
            break
        end
        s = next_s #TODO: deepcopy?
    end
    
    best_action = (length(optimal_policy) == 0) ? RockSampleAction(3) : optimal_policy[1]

    # Execute the sequence backwards to allow using the DP trick
    for i = length(optimal_policy):-1:1
        act = optimal_policy[i]
        ret *= pomdp.discount
        if act.index == 4
            rock = pomdp.rock_at_cell[cell_num(pomdp, prev_cell_coord[1], prev_cell_coord[2])+1]
            if rock != -1
                ret = expected_sampling_value[rock+1] + ret # expected sampling value is an array
            end
            continue
        end

        # Move in the opposite direction since we're going backwards
        if act.index == 0
            prev_cell_coord[1] += 1
        elseif act.index == 1
            prev_cell_coord[1] -= 1
        elseif act.index == 2
            prev_cell_coord[2] -= 1
        elseif act.index == 3
            prev_cell_coord[2] += 1
        else
            @assert(false)
        end
    end
    return ret, best_action
end
