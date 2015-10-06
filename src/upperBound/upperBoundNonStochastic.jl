import DESPOT:
    upper_bound,
    init_upper_bound

type UpperBoundNonStochastic <: DESPOTUpperBound

    upper_bound_act::Vector{Int64}
    upper_bound_memo::Vector{Float64}
    
    # Constructor
    function UpperBoundNonStochastic(pomdp::POMDP)
    
        this = new()
        #next_level_ub_memo = rs.upper_bound_memo # establish next_level_ub_memo as an alias to rs.upper_bound_memo
        
        # this executes just once per problem run
        this.upper_bound_act = Array(Int64, n_states(pomdp))    # upper_bound_act
        fill!(this.upper_bound_act, 0)
        this.upper_bound_memo = Array(Float64, n_states(pomdp)) # upper_bound_memo

    #     current_level_ub_memo = Array(Float64, n_states(pomdp))
    #     fill!(current_level_ub_memo, -Inf)
    # 
    #     next_level_ub_memo = [fringe_upper_bound(pomdp, s, config) for s = 0:n_states(pomdp)-1]
    # 
    #     for i in 1:config.search_depth # length of horizon
    #         #for s in 0:pomdp.problem.nStates-1
    #         for s in states(pomdp)
    #             #for a = 0:pomdp.problem.nActions-1
    #             for a in actions(pomdp)
    #                 #println("current state: $s")
    #                 next_state, r = step(pomdp, s, a) # TODO: change this
    #                 #println("next state: $next_state")
    #                 possibly_improved_value = r + config.discount * next_level_ub_memo[next_state+1] # TODO: move discount into pomdp
    #                 if (possibly_improved_value > current_level_ub_memo[s+1])
    #                     current_level_ub_memo[s+1] = possibly_improved_value
    #                     if i == config.search_depth
    #                         # Set best actions when last level is being computed
    #                         pomdp.upper_bound_act[s+1] = a #TODO: move to solver?
    #                     end
    #                 end
    #             end # for a
    #         end #for s
    #         
    #         # swap array references
    #         tmp = current_level_ub_memo
    #         current_level_ub_memo = next_level_ub_memo
    #         next_level_ub_memo = tmp
    # 
    #         fill!(current_level_ub_memo,-Inf)
    #     end
    # 
    #     #TODO: this can probably be done more optimally (by referencing rs.upper_bound_memo to start with),
    #     # however, this only runs once per problem and is probably not a big deal. Leave it as is for now.
    #     copy!(pomdp.upper_bound_memo, next_level_ub_memo) #TODO: move upper bound to solver or reference from this struct?
    return this
  end
end

function init_upper_bound(ub::UpperBoundNonStochastic,
                    pomdp::POMDP,
                    config::DESPOTConfig)
                          
    current_level_ub_memo = Array(Float64, n_states(pomdp))
    fill!(current_level_ub_memo, -Inf)

    next_level_ub_memo = [fringe_upper_bound(pomdp, s) for s = 0:n_states(pomdp)-1]
#    println("next_level_ub_memo: $(next_level_ub_memo[n_states(pomdp)-10:end])")
#     for j in 1:length(next_level_ub_memo) # this seems to be ok
#         println("next_level_ub_memo[$j]: $(next_level_ub_memo[j])")
#     end
#     println("rock at cell: $(pomdp.rock_at_cell)")
#     println("next_level_ub_memo[172+1]: $(next_level_ub_memo[172+1])")
    #println("discount: $(pomdp.discount)"); exit()
    #println("states: $(states(pomdp)), actions: $(actions(pomdp)), depth: $(config.search_depth)")
    
    for i in 1:config.search_depth # length of horizon
    #for i in 80:config.search_depth # length of horizon
        #for s in 0:pomdp.problem.nStates-1
        for s in states(pomdp)
            #for a = 0:pomdp.problem.nActions-1
            for a in actions(pomdp)
                #println("current state: $s")
                next_state, r = step(pomdp, s, a) # TODO: change this
#                 if i==6 && (s > 10) && (s < 15) && (a < 10)
#                     println("s: $s, a: $a, s': $next_state, next ub: $(next_level_ub_memo[next_state+1]), r=$r")
#                 end
#                     if i==2 && s==236 && a==0
#                     if i < 3 && s==172
#                         println("i:$i, s:$s, a:$a, s':$next_state, r:$r, nlm:$(next_level_ub_memo[next_state+1])");
#                     end
                possibly_improved_value = r + pomdp.discount * next_level_ub_memo[next_state+1]
                
#                 if i==80
#                     println("possiblyImprovedValue: $possibly_improved_value")
#                 end
                if (possibly_improved_value > current_level_ub_memo[s+1])
#                     if i<3 && (s==172 || s==236) #&& a==0
#                         println("piv: i=$i, s:$s, a:$a, v:$possibly_improved_value, m: $(current_level_ub_memo[s+1])");
#                     end
                    current_level_ub_memo[s+1] = possibly_improved_value
#                     if i==2 && s==236 && a==0
#                         println("s:$s, a:$a: memo: $(current_level_ub_memo[s+1])");
#                     end
                    if i == config.search_depth
                        # Set best actions when last level is being computed
                        ub.upper_bound_act[s+1] = a
                        #upper_bound_act[s+1] = a
                        #println("ub.upper_bound_act[$(s+1)]: $(ub.upper_bound_act[s+1])");
                    end
                end
            end # for a
#             if i==90
#                 exit()
#             end
        end #for s
        
        #println("current_level_ub_memo: $(current_level_ub_memo[1:10])");
        # swap array references

        tmp = current_level_ub_memo
        current_level_ub_memo = next_level_ub_memo
        next_level_ub_memo = tmp
        
#         if i < 3
#             println("------------------------------------------")
#             println("tmp[$i]: $(tmp[173])")
#             println("------------------------------------------")
#             println("current[$i]: $(current_level_ub_memo[173])")
#             println("------------------------------------------")
#             println("next_level[$i]: $(next_level_ub_memo[173])")
#             println("------------------------------------------")
#         end
            
#         if (i<6)
#             println("current_level_ub_memo $i: $(current_level_ub_memo[200:205])");# exit()
#         end
        fill!(current_level_ub_memo,-Inf)
    end

    #TODO: this can probably be done more optimally (by referencing rs.upper_bound_memo to start with),
    # however, this only runs once per problem and is probably not a big deal. Leave it as is for now.
    copy!(ub.upper_bound_memo, next_level_ub_memo)
    #println("upper_bound_memo: $(ub.upper_bound_memo[10:20])"); exit()
    
    #println("ub_actions in UB: $(ub.upper_bound_act[1:10])")
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
