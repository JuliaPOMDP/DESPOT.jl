using DESPOT
import DESPOT: upper_bound

type UpperBoundNonStochastic <: DESPOTUpperBound

    upperBoundAct::Array{Int64,1}
    upperBoundMemo::Array{Float64,1}
    
    # Constructor
    function UpperBoundNonStochastic(pomdp::DESPOTPomdp) #TODO: see if we need streams here
    
    this = new()
    #nextLevelUbMemo = rs.upperBoundMemo # establish nextLevelUbMemo as an alias to rs.upperBoundMemo
    
    # this executes just once per problem run
    this.upperBoundAct = Array(Int64, pomdp.problem.nStates)    # upperBoundAct
    fill!(this.upperBoundAct, 0)
    this.upperBoundMemo = Array(Float64, pomdp.problem.nStates) # upperBoundMemo
    currentLevelUbMemo = Array(Float64, pomdp.problem.nStates)
    fill!(currentLevelUbMemo, -Inf)

    nextLevelUbMemo = [fringeUpperBound(pomdp.problem, s, pomdp.config) for s = 0:pomdp.problem.nStates-1]

    for i in 1:pomdp.config.searchDepth # length of horizon
        for s in 0:pomdp.problem.nStates-1
            for a = 0:pomdp.problem.nActions-1
                nextState, r = step(pomdp.problem, s, a)
                possiblyImprovedValue = r + pomdp.config.discount * nextLevelUbMemo[nextState+1]
                if (possiblyImprovedValue > currentLevelUbMemo[s+1])
                    currentLevelUbMemo[s+1] = possiblyImprovedValue
                    if i == pomdp.config.searchDepth
                        # Set best actions when last level is being computed
                        pomdp.problem.upperBoundAct[s+1] = a
                    end
                end
            end # for a
        end #for s
        
        # swap array references
        tmp = currentLevelUbMemo
        currentLevelUbMemo = nextLevelUbMemo
        nextLevelUbMemo = tmp

        fill!(currentLevelUbMemo,-Inf)
    end

    #TODO: this can probably be done more optimally (by referencing rs.upperBoundMemo to start with),
    # however, this only runs once per problem and is probably not a big deal. Leave it as is for now.
    copy!(pomdp.problem.upperBoundMemo, nextLevelUbMemo)
    return this
  end
end

function upper_bound(pomdp::DESPOTPomdp, particles::Vector{DESPOTParticle})
  ws = 0.
  totalCost = 0.

  for p in particles
    ws += p.wt
    totalCost += p.wt * pomdp.problem.upperBoundMemo[p.state+1]
  end
  return totalCost / ws
end
