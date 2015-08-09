# 
#   This type maintains the current state of the world and steps it forward
#   whenever the agent takes an action and receives an observation.
#

type World
    state::Int64
    initialState::Int64
    seed::Uint32
    initialSeed::Uint32
    rewards::Array{Float64,1}

     # default constructor
     function World (problem::DESPOTProblem, seed::Uint32)
          this = new()
          this.state        = start_state(problem)      # state
          this.initialState = this.state                # initialState
          this.seed         = seed                      # seed
          this.initialSeed  = seed                      # initialSeed
          this.rewards      = Array(Float64, 0)         # rewards          
          return this
     end
end

  # Resets the world to have the same initial state and seed so that
  # a sequence of updates can be reproduced exactly.
function resetWorld(world::World)
    state = initialState
    seed = initialSeed
    rewards = 0
end

function undiscountedReturn(world::World)
    return sum(world.rewards)
end

function discountedReturn(world::World, config::Config)
    result = 0
    multiplier = 1

    for r in world.rewards
        result += multiplier * r
        multiplier *= config.discount
    end
    return result
end

# Advances the current state of the world
function step(world::World, problem::DESPOTProblem, action::Int64, config::Config)

    if OS_NAME == :Linux
        seed = Cuint[world.seed]
        randomNumber = ccall((:rand_r, "libc"), Int, (Ptr{Cuint},), seed) / config.randMax
    else #Windows, etc
        srand(world.seed)
        randomNumber = rand()
    end
    
    nextState, reward, obs = step(problem, world.state, randomNumber, action)
    world.state = nextState
    println("Action = $action")
    println("State = $nextState"); printState(problem, nextState)
    print  ("Observation = "); printObs(problem, obs)
    println("Reward = $reward")
    push!(world.rewards, reward)
    return obs, reward
end
