import POMDPs:
       create_state,
       create_action,
       create_observation,
       create_belief,
       create_transition_distribution,
       create_observation_distribution,
       transition,
       observation,
       reward,
       initial_belief,
       isterminal,
       rand!
       
import Base:
       print

typealias RockSampleState       Int64
typealias RockSampleAction      Int64
typealias RockSampleObservation Int64
typealias RockSampleReward      Float64

type RockSampleTransitionDistribution <: POMDPs.AbstractDistribution
    pomdp::POMDP
    state::RockSampleState
    action::RockSampleAction
end

function create_transition_distribution(pomdp::POMDP)
    return RockSampleTransitionDistribution(pomdp, -1, -1)
end

type RockSampleObservationDistribution <: POMDPs.AbstractDistribution
    pomdp::POMDP
    state::RockSampleState
    action::RockSampleAction
    next_state::RockSampleState
    debug::Int64 #TODO: consider removing
    
    #TODO: consider removing, not really needed except for debugging
    function RockSampleObservationDistribution(pomdp::POMDP,
                                               state::RockSampleState,
                                               action::RockSampleAction,
                                               next_state::RockSampleState,
                                               debug::Int64 = 0)
        this = new()
        this.pomdp = pomdp
        this.state = state
        this.action = action
        this.next_state = next_state
        this.debug = debug
        
        return this
    end
end

function create_observation_distribution(pomdp::POMDP)
    return RockSampleObservationDistribution(pomdp, -1, -1, -1)
end

type RockSample <: POMDPs.POMDP
    #problem parameters
    grid_size::Int64
    n_rocks::Int64
    seed::Uint32 # random seed to construct arbitrary size scenarios
    
    #problem properties
    n_cells::Int64
    n_actions::Int64
    n_states::Int64
    n_observations::Int64
    robot_start_cell::Int64
    half_eff_distance::Int64
    discount::Float64
    
    #internal variables and structures
    rock_set_start::Int64
    rock_at_cell::Array{Int64,1}
    cell_to_coords::Array{Vector{Int64},1}
    observation_effectiveness::Array{Float64,2}
    rocks::Array{Int64,1}
    T::Array{RockSampleState,2}
    R::Array{RockSampleReward,2}
    actions::Array{Int64,1} # needed for large problems
    
    #observation aliases
    #TODO: think how to best convert to const (or just wait for immutable fields to be implemented...)
    BAD_OBS::Int64
    GOOD_OBS::Int64
    NONE_OBS::Int64
    TERMINAL_OBS::Int64
    
    # the const value for 'seed' is meant to provide compatibility with the C++ version of DESPOT
    function RockSample(grid_size::Int64 = 4,
                        n_rocks::Int64 = 4;
                        seed::Uint32 = convert(Uint32, 476), #TODO: ugly, fix this
                        discount::Float64 = 0.95)
                
          this = new()
          # problem parameters
          this.grid_size = grid_size
          this.n_rocks = n_rocks   
          this.seed = seed
          this.discount = discount
           
          # problem properties
          this.n_cells = grid_size*grid_size
          this.n_actions = n_rocks + 5                   
          this.n_states = (grid_size*grid_size+1)*(1 << n_rocks)
          this.n_observations = 4
          this.robot_start_cell = 1
          this.half_eff_distance = 20
              
          #internal variables and structures
          this.rock_set_start = 0                
          this.rock_at_cell = Array(Int64, this.n_cells)
          this.cell_to_coords = Array(Vector{Int64}, this.n_cells)
          this.observation_effectiveness = Array(Float64, this.n_cells, this.n_cells)
          this.rocks = Array(Int64, this.n_rocks)                       # locations              
          this.T = Array(Int64, this.n_states, this.n_actions)
          this.R = Array(Float64, this.n_states, this.n_actions)
          this.actions = [1:n_rocks + 5] # default ordering
          this.BAD_OBS      = 0
          this.GOOD_OBS     = 1
          this.NONE_OBS     = 2
          this.TERMINAL_OBS = 3
          
          init_problem(this)
          return this
     end
end

# This function returns the start state, serving two purposes simultaneously
function create_state(pomdp::RockSample)
   return make_state(pomdp, pomdp.robot_start_cell, pomdp.rock_set_start);
end

function create_action(pomdp::RockSample)
    return -1
end

function create_observation(pomdp::RockSample)
    return -1
end

# Creates a default belief structure to store the problem's initial belief
function create_belief(pomdp::RockSample)
    return ParticleBelief{RockSampleState}(Array(Particle{RockSampleState},0))
end

function initial_belief(pomdp::RockSample,
                        belief::ParticleBelief{RockSampleState} = create_belief(pomdp))

    fill_initial_belief_particles!(pomdp, belief.particles)
    return belief
end

function initial_belief(pomdp::RockSample, belief::DESPOT.DESPOTBelief{RockSampleState})
    
    fill_initial_belief_particles!(pomdp, belief.particles)
    return belief
end

function fill_initial_belief_particles!(pomdp::RockSample, particles::Vector{Particle{RockSampleState}})    
    
    n_particles = length(particles)
    pool = Array(Particle{RockSampleState},0)   
    
    p = 1.0/(1 << pomdp.n_rocks)
    for k = 0:(1 << pomdp.n_rocks)-1 #TODO: can make faster, potentially
        push!(pool, Particle{RockSampleState}(make_state(pomdp, pomdp.robot_start_cell, k), k, p))
    end
    
    #TODO: this should not really be here, but can't think of a better place until belief is fixed
    #shuffle!(belief.particles) #TODO: Uncomment!!!

   sample_particles!(particles,
                     pool,
                     n_particles,
                     convert(Uint32, 42 $ (n_particles+1)), #TODO: fix this
                     2147483647) #TODO: fix this
    return nothing
end

# accessor functions
 function n_states(pomdp::RockSample)
    return pomdp.n_states
end

function n_actions(pomdp::RockSample)
    return pomdp.n_actions
end

function n_observations(pomdp::RockSample)
    return pomdp.n_observations
end

function states(pomdp::RockSample)
    return 0:pomdp.n_states-1 # 0-based indexing
end

function actions(pomdp::RockSample)
    return 0:pomdp.n_actions-1 # 0-based indexing
end

function observations(pomdp::RockSample)
    return 0:pomdp.n_observations-1 # 0-based indexing
end

function discount(pomdp::RockSample)
    return pomdp.discount
end

function init_4_4(pomdp::RockSample)
  pomdp.rocks[1] = cell_num(pomdp,0,2) # rocks is an array
  pomdp.rocks[2] = cell_num(pomdp,2,2)
  pomdp.rocks[3] = cell_num(pomdp,3,2)
  pomdp.rocks[4] = cell_num(pomdp,3,3)
  pomdp.robot_start_cell = cell_num(pomdp,2,0)
end

function init_7_8(pomdp::RockSample)
  pomdp.rocks[1] = cell_num(pomdp,0,1)
  pomdp.rocks[2] = cell_num(pomdp,1,5)
  pomdp.rocks[3] = cell_num(pomdp,2,2)
  pomdp.rocks[4] = cell_num(pomdp,2,3)
  pomdp.rocks[5] = cell_num(pomdp,3,6)
  pomdp.rocks[6] = cell_num(pomdp,5,0)
  pomdp.rocks[7] = cell_num(pomdp,5,3)
  pomdp.rocks[8] = cell_num(pomdp,6,2)
  pomdp.robot_start_cell = cell_num(pomdp,3,0)
end

function init_11_11(pomdp::RockSample)
  pomdp.rocks[1] = cell_num(pomdp,7,0)
  pomdp.rocks[2] = cell_num(pomdp,3,0)
  pomdp.rocks[3] = cell_num(pomdp,2,1)
  pomdp.rocks[4] = cell_num(pomdp,6,2)
  pomdp.rocks[5] = cell_num(pomdp,7,3)
  pomdp.rocks[6] = cell_num(pomdp,2,3)
  pomdp.rocks[7] = cell_num(pomdp,7,4)
  pomdp.rocks[8] = cell_num(pomdp,2,5)
  pomdp.rocks[9] = cell_num(pomdp,9,6)
  pomdp.rocks[10] = cell_num(pomdp,7,9)
  pomdp.rocks[11] = cell_num(pomdp,1,9)
  pomdp.robot_start_cell = cell_num(pomdp,5,0)
end

function init_15_15(pomdp::RockSample)
  pomdp.rocks[1] = cell_num(pomdp,7,0)
  pomdp.rocks[2] = cell_num(pomdp,3,0)
  pomdp.rocks[3] = cell_num(pomdp,2,1)
  pomdp.rocks[4] = cell_num(pomdp,6,2)
  pomdp.rocks[5] = cell_num(pomdp,7,3)
  pomdp.rocks[6] = cell_num(pomdp,2,3)
  pomdp.rocks[7] = cell_num(pomdp,7,4)
  pomdp.rocks[8] = cell_num(pomdp,2,5)
  pomdp.rocks[9] = cell_num(pomdp,9,6)
  pomdp.rocks[10] = cell_num(pomdp,7,9)
  pomdp.rocks[11] = cell_num(pomdp,1,9)
  pomdp.rocks[12] = cell_num(pomdp,8,11)
  pomdp.rocks[13] = cell_num(pomdp,10,13)
  pomdp.rocks[14] = cell_num(pomdp,9,14)
  pomdp.rocks[15] = cell_num(pomdp,2,12)
  pomdp.robot_start_cell = cell_num(pomdp,5,0)
end

function start_state(pomdp::RockSample)
   return make_state(pomdp, pomdp.robot_start_cell, pomdp.rock_set_start);
end

function init_general(pomdp::RockSample, seed::Array{Uint32,1})
  
    rockIndex::Int64 = 1 # rocks is an array
    if OS_NAME != :Linux
        srand(seed[1])
    end
    
    while rockIndex <= pomdp.n_rocks
        if OS_NAME == :Linux
            cell = ccall((:rand_r, "libc"), Int, (Ptr{Cuint},), seed) % pomdp.n_cells
        else
            cell = rand(0:pomdp.config.rand_max) % pomdp.n_cells 
        end
        
        if findfirst(pomdp.rocks, cell) == 0
            pomdp.rocks[rockIndex] = cell
            rockIndex += 1
        end
    end
    pomdp.robot_start_cell = cell_num(pomdp, iround(pomdp.grid_size/2), 0)
end

function init_problem (pomdp::RockSample)

    pomdp.rocks = Array(Int64, pomdp.n_rocks)
    seed = Cuint[convert(Uint32, pomdp.seed)]
    
    if pomdp.grid_size == 4 && pomdp.n_rocks == 4
        init_4_4(pomdp)
    elseif pomdp.grid_size == 7 && pomdp.n_rocks == 8
        init_7_8(pomdp)
    elseif pomdp.grid_size == 11 && pomdp.n_rocks == 11
        init_11_11(pomdp)
    elseif pomdp.grid_size == 15 && pomdp.n_rocks == 15
        init_15_15(pomdp)
    else
        init_general(pomdp, seed)
    end
  
    # Compute rock set start
    pomdp.rock_set_start = 0

    if OS_NAME != :Linux
        srand(seed[1])
    end
  
    for i in 0 : pomdp.n_rocks-1
        if OS_NAME == :Linux
            rand_num = ccall((:rand_r, "libc"), Int, (Ptr{Cuint},), seed)
        else #Windows, etc
            rand_num = rand(0:pomdp.config.rand_max)
        end

        if (rand_num & 1) == 1
            pomdp.rock_set_start |= (1 << i)
        end
    end

    # Fill in cellToCoord and init rock_at_cell mappings
    fill!(pomdp.rock_at_cell, -1)
    
    for i in 0 : pomdp.n_cells-1
        pomdp.cell_to_coords[i+1] = [itrunc(i/pomdp.grid_size), i % pomdp.grid_size]
    end

    for i in 0 : pomdp.n_rocks-1
        pomdp.rock_at_cell[pomdp.rocks[i+1]+1] = i # rock_at_cell and rocks are arrays
    end

    # T and R - ALL INDICES BELOW ARE OFFSET BY +1 (for 1-based array indexing)
    
    for cell in 0 : pomdp.n_cells-1
        for rock_set = 0:(1 << pomdp.n_rocks)-1
        s = make_state(pomdp, cell, rock_set)
        
            #initialize transition and rewards with default values
            for a in 0:pomdp.n_actions-1
                pomdp.T[s+1,a+1] = s
                pomdp.R[s+1,a+1] = 0.
            end
            
            row, col = pomdp.cell_to_coords[cell+1]
            # North
            if row == 0
                pomdp.T[s+1,1] = s
                pomdp.R[s+1,1] = -100.
            else
                pomdp.T[s+1,1] = make_state(pomdp, cell_num(pomdp,row-1,col),rock_set)
                pomdp.R[s+1,1] = 0
            end

            # South
            if row == pomdp.grid_size-1
                pomdp.T[s+1,2] = s
                pomdp.R[s+1,2] = -100.
            else
                pomdp.T[s+1,2] = make_state(pomdp, cell_num(pomdp,row+1,col), rock_set)
                pomdp.R[s+1,2] = 0.
            end

            # East
            if col == pomdp.grid_size-1
                pomdp.T[s+1,3] = make_state(pomdp, pomdp.n_cells, rock_set)
                pomdp.R[s+1,3] = 10.
            else
                pomdp.T[s+1,3] = make_state(pomdp, cell_num(pomdp,row,col+1), rock_set)
                pomdp.R[s+1,3] = 0.
            end

            # West
            if col == 0
                pomdp.T[s+1,4] = s
                pomdp.R[s+1,4] = -100.
            else
                pomdp.T[s+1,4] = make_state(pomdp, cell_num(pomdp,row,col-1), rock_set)
                pomdp.R[s+1,4] = 0.
            end

            # Sample
            rock = pomdp.rock_at_cell[cell+1] # array
            if rock != -1
                if rock_status(rock, rock_set)
                    pomdp.T[s+1,5] = make_state(pomdp, cell, sample_rock_set(rock, rock_set));
                    pomdp.R[s+1,5] = +10.;
                else
                    pomdp.T[s+1,5] = s
                    pomdp.R[s+1,5] = -10.
                end
            else
                pomdp.T[s+1,5] = s
                pomdp.R[s+1,5] = -100.
            end

            # Check
            for a in 5:pomdp.n_actions-1
                pomdp.T[s+1,a+1] = s
                pomdp.R[s+1,a+1] = 0.
            end
        end
    end

    # Terminal states
    for k = 0:(1 << pomdp.n_rocks)-1
        s = make_state(pomdp, pomdp.n_cells, k);
        for a in 0:pomdp.n_actions-1
        pomdp.T[s+1,a+1] = s
        pomdp.R[s+1,a+1] = 0.
        end
    end

    # precompute observation effectiveness table
    for i in 0 : pomdp.n_cells-1
        for j in 0 : pomdp.n_cells-1
        agent = pomdp.cell_to_coords[i+1]
        other = pomdp.cell_to_coords[j+1]
        dist = sqrt((agent[1] - other[1])^2 + (agent[2]-other[2])^2)
        pomdp.observation_effectiveness[i+1,j+1] = (1 + 2^(-dist / pomdp.half_eff_distance)) * 0.5 # Array indexing starts from 1.
                                                                                    # Remember to subtract one to go back
        end
    end
end

# True for good rock, false for bad rock, x can be a rock set or state
function rock_status(rock::Int64, x::Int64)
    return (((x >>> rock) & 1) == 1 ? true : false)
end

function cell_num(pomdp::RockSample, row::Int64, col::Int64)
    return row * pomdp.grid_size + col
end

function make_state(pomdp::RockSample, cell::Int64, rock_set::Int64)
    return convert(Int64, (cell << pomdp.n_rocks) + rock_set)
end

function reward(pomdp::RockSample,
                state::RockSampleState,
                action::RockSampleAction)

    return pomdp.R[state+1, action+1]
end

function transition(pomdp::RockSample,
                    state::RockSampleState,
                    action::RockSampleAction,
                    distribution::RockSampleTransitionDistribution =
                                create_transition_distribution(pomdp))
    distribution.pomdp = pomdp
    distribution.state = state
    distribution.action = action

    return nothing
end

function observation(pomdp::RockSample,
                     state::RockSampleState,
                     action::RockSampleAction,
                     next_state::RockSampleState,
                     distribution::RockSampleObservationDistribution =
                                create_observation_distribution(pomdp))
    distribution.pomdp = pomdp
    distribution.state = next_state    
    distribution.action = action
    distribution.next_state = next_state

    return nothing
end

# TODO: see if there is a way to clean-up the interface and pass sample by reference
function rand!(rng::AbstractRNG,
               sample::RockSampleState,
               distribution::RockSampleTransitionDistribution)
    
    sample = distribution.pomdp.T[distribution.state+1, distribution.action+1]
    return sample
end

function rand!(rng::AbstractRNG,
               sample::RockSampleObservation,
               distribution::RockSampleObservationDistribution)
    
    # generate a new random number regardless of whether it's used below or not
    rand_num = rand!(rng)
    
    if (distribution.action < 5)
        sample = isterminal(distribution.pomdp, distribution.next_state) ?
                    distribution.pomdp.TERMINAL_OBS : distribution.pomdp.NONE_OBS # rs.T is an array
    else
        rock_cell = distribution.pomdp.rocks[distribution.action - 4] # would be [action-5] with 0-based indexing
        agent_cell = cell_of(distribution.pomdp, distribution.state)
        eff = distribution.pomdp.observation_effectiveness[agent_cell+1, rock_cell+1]
        
        if (rand_num <= eff) == rock_status(distribution.action - 5, distribution.state)   
            sample = distribution.pomdp.GOOD_OBS
        else
            sample = distribution.pomdp.BAD_OBS
        end
    end
    
    return sample
end

function pdf(distribution::RockSampleObservationDistribution, obs::RockSampleObservation)
  # Terminal state should match terminal obs
  if isterminal(distribution.pomdp, distribution.next_state)
      if obs == distribution.pomdp.TERMINAL_OBS
          return 1.
      else
          return 0.
      end
  end

  if (distribution.action < 5)
      if obs == distribution.pomdp.NONE_OBS
          return 1.
      else
          return 0.
      end
  end

  if ((obs != distribution.pomdp.GOOD_OBS) && (obs != distribution.pomdp.BAD_OBS))
    return 0.
  end

  rock = distribution.action - 5
  rockCell = distribution.pomdp.rocks[rock+1]
  agentCell = cell_of(distribution.pomdp, distribution.next_state)

  eff = distribution.pomdp.observation_effectiveness[agentCell+1, rockCell+1]
  
  rstatus = rock_status(rock, distribution.next_state)
  if ((obs == distribution.pomdp.GOOD_OBS) && (rstatus == true)) ||
     ((obs == distribution.pomdp.BAD_OBS) && (rstatus == false)) 
    return eff
  else
    return 1. - eff
  end
end

# TODO: this should work, but does not for some reason
#function isTerminal(s)
#  return (cell_of(s) == n_cells)
#end

function isterminal(pomdp::RockSample, s::RockSampleState)
  if cell_of(pomdp,s) == pomdp.n_cells
    return true
  else
    return false
  end
end

# Which cell the agent is in
function cell_of(pomdp::RockSample, s::RockSampleState)
  return s >>> pomdp.n_rocks
end

# The rock set after sampling a rock from it
function sample_rock_set(rock::Int64, rock_set::Int64)
  return rock_set & ~(1 << rock)
end

# The set of rocks in the state
function rock_set_of(pomdp::RockSample, s::RockSampleState)
    return s & ((1 << pomdp.n_rocks)-1)
end

function show_state(pomdp::RockSample, s::RockSampleState)
  ac = cell_of(pomdp, s)
  for i in 0:pomdp.grid_size-1
    for j in 0:pomdp.grid_size-1
      if ac == cell_num(pomdp,i,j)
        if pomdp.rock_at_cell[ac+1] == -1 # array
          print("R ")
        elseif rock_status(pomdp.rock_at_cell[ac+1], rock_set_of(pomdp,s))
          print("G ")
        else
          print("B ")
        end # if rock_at_cell[ac] == -1
        continue
      end # if ac == cell_num(i, j)
      if pomdp.rock_at_cell[cell_num(pomdp,i,j)+1] == -1
        print(". ")
      elseif (rock_status(pomdp.rock_at_cell[cell_num(pomdp,i,j)+1], rock_set_of(pomdp,s)))
        print("1 ")
      else
        print("0 ")
      end # if rock_at_cell[cell_num(i,j)] == -1

    end # for j in 1:grid_size
    println("")
  end # i in 1:grid_size
end

function random_state(pomdp::RockSample, seed::Uint32)
    cseed = Cuint[seed]
    ccall((:srand, "libc"), Void, (Ptr{Cuint},), cseed)
    random_number = ccall((:rand, "libc"), Int, (),)
    return random_number % pomdp.n_states
end

function show_obs(pomdp::RockSample, obs::RockSampleObservation)
    if obs == pomdp.NONE_OBS
        println("NONE")
    elseif obs == pomdp.GOOD_OBS
        println("GOOD")
    elseif obs == pomdp.BAD_OBS
        println("BAD")
    elseif obs == pomdp.TERMINAL_OBS
        println("TERMINAL")
    else
        println("UNKNOWN")
    end
end

#TODO: hack! Only needed until belief is fixed
function sample_particles(pool::Vector,
                          N::Int64,
                          seed::Uint32,
                          rand_max::Int64)

    sampled_particles = Array(Particle, 0)

    # Ensure particle weights sum to exactly 1
    sum_without_last =  0;
    
    for i in 1:length(pool)-1
        sum_without_last += pool[i].weight
    end
    
    end_weight = 1 - sum_without_last

    # Divide the cumulative frequency into N equally-spaced parts
    num_sampled = 0
    
    if OS_NAME == :Linux
        cseed = Cuint[seed]
        r = ccall((:rand_r, "libc"), Int, (Ptr{Cuint},), cseed)/rand_max/N
    else #Windows, etc
        srand(seed)
        r = rand()/N
    end

    curr_particle = 0
    cum_sum = 0
    while num_sampled < N
        while cum_sum < r
            curr_particle += 1
            if curr_particle == length(pool)
                cum_sum += end_weight
            else
                cum_sum += pool[curr_particle].weight
            end
        end

        new_particle = Particle(pool[curr_particle].state, 1.0 / N)
        push!(sampled_particles, new_particle)
        num_sampled += 1
        r += 1.0 / N
    end
    return sampled_particles
end
