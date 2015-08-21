using DESPOT
import DESPOT: 
              start_state,
              initial_belief,
              isterminal,
              lower_bound,
              init_problem,
              step,
              display_state,
              display_obs,
              obs_probability,
              random_state

type RockSample <: DESPOTProblem
    #problem parameters
    gridSize::Int64
    nRocks::Int64
    
    #problem properties
    nCells::Int64
    nActions::Int64
    nStates::Int64
    nObservations::Int64
    robotStartCell::Int64
    halfEffDistance::Int64
    
    #internal variables and structures
    rockSetStart::Int64
    weightSumOfState::Array{Float64,1}
    upperBoundAct::Array{Int64,1}
    rockAtCell::Array{Int64,1}
    cellToCoords::Array{Vector{Int64},1}
    observationEffectiveness::Array{Float64,2}
    upperBoundMemo::Array{Float64,1}
    rocks::Array{Int64,1}
    T::Array{Int64,2}
    R::Array{Float64,2}
    actions::Array{Int64,1} # needed for large problems
    
    #observation aliases
    #TODO: think how to best convert to const (or just wait for immutable fields to be implemented...)
    kBad::Int64
    kGood::Int64
    kNone::Int64
    kTerminal::Int64
    

    function RockSample(gridSize::Int64 = 4, nRocks::Int64 = 4)
                
          this = new()
          # problem parameters
          this.gridSize = gridSize
          this.nRocks = nRocks                              
           
          # problem properties
          this.nCells = gridSize*gridSize

          this.nActions = nRocks + 5                   
          this.nStates = (gridSize*gridSize+1)*(1 << nRocks)
          this.nObservations = 4
          this.robotStartCell = 1
          this.halfEffDistance = 20
              
          #internal variables and structures
          this.rockSetStart = 0               
          this.weightSumOfState = Array(Float64,this.nStates)  
          this.upperBoundAct = Array(Int64,this.nStates)        
          this.rockAtCell = Array(Int64, this.nCells)
          this.cellToCoords = Array(Vector{Int64}, this.nCells)
          this.observationEffectiveness = Array(Float64, this.nCells, this.nCells)
          this.upperBoundMemo = Array(Float64,(gridSize*gridSize+1)*(1 << nRocks))
          this.rocks = Array(Int64, this.nRocks)                       # locations              
          this.T = Array(Int64, this.nStates, this.nActions)
          this.R = Array(Float64, this.nStates, this.nActions)
          this.actions = [1:nRocks + 5] # default ordering
          this.kBad      = 0
          this.kGood     = 1
          this.kNone     = 2
          this.kTerminal = 3
          
          return this
     end
end

function init_4_4(problem::RockSample)
  problem.rocks[1] = cellNum(problem,0,2) # rocks is an array
  problem.rocks[2] = cellNum(problem,2,2)
  problem.rocks[3] = cellNum(problem,3,2)
  problem.rocks[4] = cellNum(problem,3,3)
  problem.robotStartCell = cellNum(problem,2,0)
end

function init_7_8(problem::RockSample)
  problem.rocks[1] = cellNum(problem,0,1)
  problem.rocks[2] = cellNum(problem,1,5)
  problem.rocks[3] = cellNum(problem,2,2)
  problem.rocks[4] = cellNum(problem,2,3)
  problem.rocks[5] = cellNum(problem,3,6)
  problem.rocks[6] = cellNum(problem,5,0)
  problem.rocks[7] = cellNum(problem,5,3)
  problem.rocks[8] = cellNum(problem,6,2)
  problem.robotStartCell = cellNum(problem,3,0)
end

function init_11_11(problem::RockSample)
  problem.rocks[1] = cellNum(problem,7,0)
  problem.rocks[2] = cellNum(problem,3,0)
  problem.rocks[3] = cellNum(problem,2,1)
  problem.rocks[4] = cellNum(problem,6,2)
  problem.rocks[5] = cellNum(problem,7,3)
  problem.rocks[6] = cellNum(problem,2,3)
  problem.rocks[7] = cellNum(problem,7,4)
  problem.rocks[8] = cellNum(problem,2,5)
  problem.rocks[9] = cellNum(problem,9,6)
  problem.rocks[10] = cellNum(problem,7,9)
  problem.rocks[11] = cellNum(problem,1,9)
  problem.robotStartCell = cellNum(problem,5,0)
end

function init_15_15(problem::RockSample)
  problem.rocks[1] = cellNum(problem,7,0)
  problem.rocks[2] = cellNum(problem,3,0)
  problem.rocks[3] = cellNum(problem,2,1)
  problem.rocks[4] = cellNum(problem,6,2)
  problem.rocks[5] = cellNum(problem,7,3)
  problem.rocks[6] = cellNum(problem,2,3)
  problem.rocks[7] = cellNum(problem,7,4)
  problem.rocks[8] = cellNum(problem,2,5)
  problem.rocks[9] = cellNum(problem,9,6)
  problem.rocks[10] = cellNum(problem,7,9)
  problem.rocks[11] = cellNum(problem,1,9)
  problem.rocks[12] = cellNum(problem,8,11)
  problem.rocks[13] = cellNum(problem,10,13)
  problem.rocks[14] = cellNum(problem,9,14)
  problem.rocks[15] = cellNum(problem,2,12)
  problem.robotStartCell = cellNum(problem,5,0)
end

function start_state(problem::RockSample)
   return makeState(problem, problem.robotStartCell, problem.rockSetStart);
end

# function initial_belief(problem::RockSample)
#   stateProbabilitiesArray = Array(DESPOTStateProbability, 0)
#   p = 1.0/(1 << problem.nRocks)
#   for k = 0:(1 << problem.nRocks)-1
#     stateProbability =
#     push!(stateProbabilitiesArray, DESPOTStateProbability(makeState(problem, problem.robotStartCell, k), p))
#   end
#   return DESPOTBelief(stateProbabilitiesArray)
# end

function initial_belief(problem::RockSample)
  particles = Array(DESPOTParticle, 0)
  p = 1.0/(1 << problem.nRocks)
  for k = 0:(1 << problem.nRocks)-1
    push!(particles, DESPOTParticle(makeState(problem, problem.robotStartCell, k), 0, p))
  end
  return DESPOTBelief(particles)
end

function initGeneral(pomdp::DESPOTPomdp, seed::Array{Uint32,1})
  
    rockIndex::Int64 = 1 # rocks is an array
    if OS_NAME != :Linux
        srand(seed[1])
    end
    
    while rockIndex <= pomdp.problem.nRocks
        if OS_NAME == :Linux
            cell = ccall((:rand_r, "libc"), Int, (Ptr{Cuint},), seed) % pomdp.problem.nCells
        else
            cell = rand(0:pomdp.config.randMax) % pomdp.problem.nCells 
        end
        
        if findfirst(pomdp.problem.rocks, cell) == 0
            pomdp.problem.rocks[rockIndex] = cell
            rockIndex += 1
        end
    end
    pomdp.problem.robotStartCell = cellNum(pomdp.problem, iround(pomdp.problem.gridSize/2), 0)
end

function init_problem (pomdp::DESPOTPomdp)

    pomdp.problem.rocks = Array(Int64, pomdp.problem.nRocks)
    seed = Cuint[convert(Uint32, pomdp.randomStreams.modelSeed)]
    
    if pomdp.problem.gridSize == 4 && pomdp.problem.nRocks == 4
        init_4_4(pomdp.problem)
    elseif pomdp.problem.gridSize == 7 && pomdp.problem.nRocks == 8
        init_7_8(pomdp.problem)
    elseif pomdp.problem.gridSize == 11 && pomdp.problem.nRocks == 11
        init_11_11(pomdp.problem)
    elseif pomdp.problem.gridSize == 15 && pomdp.problem.nRocks == 15
        init_15_15(pomdp.problem)
    else
        initGeneral(pomdp, seed)
    end
  
    # Compute rock set start
    pomdp.problem.rockSetStart = 0

    if OS_NAME != :Linux
        srand(seed[1])
    end
  
    for i in 0 : pomdp.problem.nRocks-1
        if OS_NAME == :Linux
            randNumber = ccall((:rand_r, "libc"), Int, (Ptr{Cuint},), seed)
        else #Windows, etc
            randNumber = rand(0:pomdp.config.randMax)
        end

        if (randNumber & 1) == 1
            pomdp.problem.rockSetStart |= (1 << i)
        end
    end

    # initialize various structures
    fill!(pomdp.problem.weightSumOfState, -Inf)
    fill!(pomdp.problem.upperBoundAct, 0)

    # Fill in cellToCoord and init rockAtCell mappings
    fill!(pomdp.problem.rockAtCell, -1)
    
    for i in 0 : pomdp.problem.nCells-1
        pomdp.problem.cellToCoords[i+1] = [itrunc(i/pomdp.problem.gridSize), i % pomdp.problem.gridSize]
    end

    for i in 0 : pomdp.problem.nRocks-1
        pomdp.problem.rockAtCell[pomdp.problem.rocks[i+1]+1] = i # rockAtCell and rocks are arrays
    end

    # T and R - ALL INDICES BELOW ARE OFFSET BY +1 (1-based array indexing)
    #   problem.T = Array(Int64, problem.nStates, problem.nActions)
    #   problem.R = Array(Float64, problem.nStates, problem.nActions)
    
    for cell in 0 : pomdp.problem.nCells-1
        for rockSet = 0:(1 << pomdp.problem.nRocks)-1
        s = makeState(pomdp.problem, cell, rockSet)
        
            #initialize transition and rewards with default values
            for a in 0:pomdp.problem.nActions-1
                pomdp.problem.T[s+1,a+1] = s
                pomdp.problem.R[s+1,a+1] = 0.
            end
            
            row, col = pomdp.problem.cellToCoords[cell+1]
            # North
            if row == 0
                pomdp.problem.T[s+1,1] = s
                pomdp.problem.R[s+1,1] = -100.
            else
                pomdp.problem.T[s+1,1] = makeState(pomdp.problem, cellNum(pomdp.problem,row-1,col),rockSet)
                pomdp.problem.R[s+1,1] = 0
            end

            # South
            if row == pomdp.problem.gridSize-1
                pomdp.problem.T[s+1,2] = s
                pomdp.problem.R[s+1,2] = -100.
            else
                pomdp.problem.T[s+1,2] = makeState(pomdp.problem, cellNum(pomdp.problem,row+1,col), rockSet)
                pomdp.problem.R[s+1,2] = 0.
            end

            # East
            if col == pomdp.problem.gridSize-1
                pomdp.problem.T[s+1,3] = makeState(pomdp.problem, pomdp.problem.nCells, rockSet)
                pomdp.problem.R[s+1,3] = 10.
            else
                pomdp.problem.T[s+1,3] = makeState(pomdp.problem, cellNum(pomdp.problem,row,col+1), rockSet)
                pomdp.problem.R[s+1,3] = 0.
            end

            # West
            if col == 0
                pomdp.problem.T[s+1,4] = s
                pomdp.problem.R[s+1,4] = -100.
            else
                pomdp.problem.T[s+1,4] = makeState(pomdp.problem, cellNum(pomdp.problem,row,col-1), rockSet)
                pomdp.problem.R[s+1,4] = 0.
            end

            # Sample
            rock = pomdp.problem.rockAtCell[cell+1] # array
            if rock != -1
                if rockStatus(rock, rockSet)
                    pomdp.problem.T[s+1,5] = makeState(pomdp.problem, cell, sampleRockSet(rock, rockSet));
                    pomdp.problem.R[s+1,5] = +10.;
                else
                    pomdp.problem.T[s+1,5] = s
                    pomdp.problem.R[s+1,5] = -10.
                end
            else
                pomdp.problem.T[s+1,5] = s
                pomdp.problem.R[s+1,5] = -100.
            end

            # Check
            for a in 5:pomdp.problem.nActions-1
                pomdp.problem.T[s+1,a+1] = s
                pomdp.problem.R[s+1,a+1] = 0.
            end
        end
    end

    # Terminal states
    for k = 0:(1 << pomdp.problem.nRocks)-1
        s = makeState(pomdp.problem, pomdp.problem.nCells, k);
        for a in 0:pomdp.problem.nActions-1
        pomdp.problem.T[s+1,a+1] = s
        pomdp.problem.R[s+1,a+1] = 0.
        end
    end

    # precompute observation effectiveness table
    #problem.observationEffectiveness = Array(Float64, problem.nCells, problem.nCells)
    
    for i in 0 : pomdp.problem.nCells-1
        for j in 0 : pomdp.problem.nCells-1
        agent = pomdp.problem.cellToCoords[i+1]
        other = pomdp.problem.cellToCoords[j+1]
        dist = sqrt((agent[1] - other[1])^2 + (agent[2]-other[2])^2)
        pomdp.problem.observationEffectiveness[i+1,j+1] = (1 + 2^(-dist / pomdp.problem.halfEffDistance)) * 0.5 # Array indexing starts from 1.
                                                                                    # Remember to subtract one to go back
        end
    end
end

# True for good rock, false for bad rock, x can be a rock set or state
function rockStatus(rock::Int64, x::Int64)
    return (((x >>> rock) & 1) == 1 ? true : false)
end

function cellNum(problem::RockSample, row::Int64, col::Int64)
    return row * problem.gridSize + col
end

function makeState(problem::RockSample, cell::Int64, rockSet::Int64)
    return convert(Int64, (cell << problem.nRocks) + rockSet)
end

# stochastic version
function step(problem::RockSample, s::Int64, randNum::Float64, action::Int64)
  #println("state: $s")
  reward = problem.R[s+1,action+1]
  if (action < 5)
    obs = isterminal(problem, problem.T[s+1, action+1]) ? problem.kTerminal : problem.kNone # rs.T is an array
  else
    rockCell = problem.rocks[action - 4] # would be [action-5] with 0-based indexing
    agentCell = cellOf(problem,s)
    eff = problem.observationEffectiveness[agentCell+1,rockCell+1]
    if (randNum <= eff) == rockStatus(action - 5, s)
      obs = problem.kGood
    else
      obs = problem.kBad
    end
  end
   nextState = problem.T[s+1,action+1]
   return nextState, reward, obs
end

# deterministic version
function step(problem::RockSample, s::Int64, action::Int64)
  reward = problem.R[s+1,action+1]
  newState = problem.T[s+1,action+1]
  return newState, reward
end

function obs_probability(problem::RockSample, obs::Int64, s::Int64, action::Int64)
  # Terminal state should match terminal obs
  if isterminal(problem,s)
      if obs == problem.kTerminal
          return 1.
      else
          return 0.
      end
  end

  if (action < 5)
      if obs == problem.kNone
          return 1.
      else
          return 0.
      end
  end

  if ((obs != problem.kGood) && (obs != problem.kBad))
    return 0.
  end

  rock = action - 5
  rockCell = problem.rocks[rock+1]
  agentCell = cellOf(problem,s)

  eff = problem.observationEffectiveness[agentCell+1,rockCell+1]
  if (obs == rockStatus(rock, s))
    return eff
  else
    return 1. - eff
  end
end

# TODO: this should work, but does not for some reason
#function isTerminal(s)
#  return (cellOf(s) == nCells)
#end

function isterminal(problem::RockSample, s::Int64)
  if cellOf(problem,s) == problem.nCells
    return true
  else
    return false
  end
end

# Which cell the agent is in
function cellOf(problem::RockSample, s::Int64)
  return s >>> problem.nRocks
end

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

function lower_bound(problem::RockSample, particles::Vector{DESPOTParticle}, streamPosition::Int64, config::DESPOTConfig)

  stateSeen = Dict{Int64,Int64}()

  # Since for this problem the cell that the rover is in is deterministic, picking pretty much
  # any particle state is ok
  if length(particles) > 0
    if isterminal(problem, particles[1].state)
        return 0., -1 # lower bound value and best action
    end
  end

  # The expected value of sampling a rock, over all particles
  expectedSamplingValue = fill(0., problem.nRocks)
  seenPtr = 0

  # Compute the expected sampling value of each rock. Instead of factoring
  # the weight of each particle, we first record the weight of each state.
  # This is so that the inner loop that updates the expected value of each
  # rock runs once per state seen, instead of once per particle seen. If
  # there are lots of common states between particles, this gives a
  # significant speedup to the search because the lower bound is the
  # bottleneck.

  for p in particles
    if problem.weightSumOfState[p.state+1] == -Inf #Array
      problem.weightSumOfState[p.state+1] = p.wt
      stateSeen[seenPtr] = p.state
      seenPtr += 1
    else
      problem.weightSumOfState[p.state+1] += p.wt;
    end
  end
  
  ws = 0
  for i in 0:seenPtr-1
    s = stateSeen[i]
    ws += problem.weightSumOfState[s+1]
    for j in 0:problem.nRocks-1 #TODO: check for a possible bug in the original code
      expectedSamplingValue[j+1] += problem.weightSumOfState[s+1] * (rockStatus(j, s) ? 10. : -10.)
    end
  end
  
  # Reset for next use
  fill!(problem.weightSumOfState, -Inf)

  mostLikelyRockSet = 0
  for i in 0:problem.nRocks-1
    expectedSamplingValue[i+1] /= ws
    # Threshold the average to good or bad
    if expectedSamplingValue[i+1] > -config.tiny
      mostLikelyRockSet |= (1 << i)
    end
    if almost_the_same(0., expectedSamplingValue[i+1], config)
      expectedSamplingValue[i+1] = 0.
    end
  end

  # Since for this problem the cell that the rover is in is deterministic, picking pretty much
  # any particle state is ok
  mostLikelyState = makeState(problem, cellOf(problem, particles[1].state), mostLikelyRockSet)
  s = mostLikelyState

  # Sequence of actions taken in the optimal policy
  optimalPolicy = Array(Int,0)
  ret = 0.
  reward = 0.
  prevCellCoord = [0,0] # initial value - should cause error if not properly assigned
  
  #println("Most likely state: $s")
  while true
    act = problem.upperBoundAct[s+1]
    sTest, reward = step(problem, s, act) # deterministic version
    if isterminal(problem, sTest)
      prevCellCoord[1] = problem.cellToCoords[cellOf(problem, s)+1][1]
      prevCellCoord[2] = problem.cellToCoords[cellOf(problem, s)+1][2]
      ret = 10.
      break
    end
    push!(optimalPolicy, act)
    if length(optimalPolicy) == config.searchDepth
      prevCellCoord[1] = problem.cellToCoords[cellOf(problem, sTest)+1][1]
      prevCellCoord[2] = problem.cellToCoords[cellOf(problem, sTest)+1][2]
      ret = 0.
      break
    end
    s = sTest
  end
  
  bestAction = (length(optimalPolicy) == 0) ? 3 : optimalPolicy[1]

  # Execute the sequence backwards to allow using the DP trick mentioned
  # earlier.
  for i = length(optimalPolicy):-1:1
    act = optimalPolicy[i]
    ret *= config.discount
    if act == 4
      rock = problem.rockAtCell[cellNum(problem, prevCellCoord[1], prevCellCoord[2])+1]
      if rock != -1
        ret = expectedSamplingValue[rock+1] + ret # expected sampling value is an array
      end
      continue
    end

    # Move in the opposite direction since we're going backwards
      if act == 0
        prevCellCoord[1] += 1
      elseif act == 1
        prevCellCoord[1] -= 1
      elseif act == 2
        prevCellCoord[2] -= 1
      elseif act == 3
        prevCellCoord[2] += 1
      else
        @assert(false)
      end
    end
  return ret, bestAction
end

# The rock set after sampling a rock from it
function sampleRockSet(rock::Int64, rockSet::Int64)
  return rockSet & ~(1 << rock)
end

# The set of rocks in the state
function rockSetOf(problem::DESPOTProblem, s::Int64)
    return s & ((1 << problem.nRocks)-1)
end

function display_state(problem::DESPOTProblem, s::Int64)
  ac = cellOf(problem, s)
  for i in 0:problem.gridSize-1
    for j in 0:problem.gridSize-1
      if ac == cellNum(problem,i,j)
        if problem.rockAtCell[ac+1] == -1 # array
          print("R ")
        elseif rockStatus(problem.rockAtCell[ac+1], rockSetOf(problem,s))
          print("G ")
        else
          print("B ")
        end # if rockAtCell[ac] == -1
        continue
      end # if ac == cellNum(i, j)
      if problem.rockAtCell[cellNum(problem,i,j)+1] == -1
        print(". ")
      elseif (rockStatus(problem.rockAtCell[cellNum(problem,i,j)+1], rockSetOf(problem,s)))
        print("1 ")
      else
        print("0 ")
      end # if rockAtCell[cellNum(i,j)] == -1

    end # for j in 1:gridSize
    println("")
  end # i in 1:gridSize
end

function random_state(problem::DESPOTProblem, seed::Uint32)
    cseed = Cuint[seed]
    ccall((:srand, "libc"), Void, (Ptr{Cuint},), cseed)
    randomNumber = ccall((:rand, "libc"), Int, (),)
    return randomNumber % problem.nStates
end

function fringeUpperBound(problem::DESPOTProblem, s::Int64, config::DESPOTConfig)
  if isterminal(problem, s)
    return 0
  end

  rockSet = rockSetOf(problem, s)
  nGood = 0
  while rockSet != 0
    nGood += rockSet & 1
    rockSet >>>= 1
  end

  # Assume a good rock is sampled at each step and an exit is made in the last
  if config.discount < 1
    return 10. * (1 - (config.discount^(nGood+1))) / (1 - config.discount)
  else
    return 10. * (nGood + 1)
  end
end

function display_obs(problem::DESPOTProblem, obs::Int64)
    if obs == problem.kNone
        println("NONE")
    elseif obs == problem.kGood
        println("GOOD")
    elseif obs == problem.kBad
        println("BAD")
    elseif obs == problem.kTerminal
        println("TERMINAL")
    else
        println("UNKNOWN")
    end
end

