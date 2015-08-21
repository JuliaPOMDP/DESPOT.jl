type DESPOTBeliefUpdateParticle <: DESPOTBeliefUpdate
  numUpdates::Int64
  beliefUpdateSeed::Int64
  initBeliefUpdateSeed::Int64
  #default constructor
  DESPOTBeliefUpdateParticle(beliefUpdateSeed::Uint32) =
    new(
        0,                     # numUpdates
        beliefUpdateSeed,      # beliefUpdateSeed
        beliefUpdateSeed       # initBeliefUpdateSeed
    )
end

# Resets the updater to its starting seed value.
# Useful when a sequence of updates needs to be reproduced exactly.
function resetBelief(beliefUpdate::DESPOTBeliefUpdateParticle)
    beliefUpdate.beliefUpdateSeed = convert(Uint32, beliefUpdate.initBeliefUpdateSeed)
    beliefUpdate.numUpdates = 0
end

function normalize(particles::Array{DESPOTParticle,1})
  probSum = 0.
  for p in particles
    probSum += p.wt
  end
  for p in particles
    p.wt /= probSum
  end
end

# TODO: This probably can be replaced by just a weighted sampling call - check later
function sampleParticles(bu::DESPOTBeliefUpdateParticle, pool::Array{DESPOTParticle,1}, N::Uint32, config::DESPOTConfig)

    sampledParticles = Array(DESPOTParticle, 0)

    # Ensure particle weights sum to exactly 1
    sumWithoutLast = 0;
    
    for i in 1:length(pool)-1
        sumWithoutLast += pool[i].wt
    end
    
    endWeight = 1 - sumWithoutLast

    # Divide the cumulative frequency into N equally-spaced parts
    numSampled = 0
    
    if OS_NAME == :Linux
        seed = Cuint[bu.beliefUpdateSeed]
        r = ccall((:rand_r, "libc"), Int, (Ptr{Cuint},), seed)/config.randMax/N
    else #Windows, etc
        srand(bu.beliefUpdateSeed)
        r = rand()/N
    end

    currParticle = 0
    cumSum = 0
    while numSampled < N
        while cumSum < r
            currParticle += 1
            if currParticle == length(pool)
                cumSum += endWeight
            else
                cumSum += pool[currParticle].wt
            end
        end

        newParticle = DESPOTParticle(pool[currParticle].state, numSampled, 1.0 / N)
        push!(sampledParticles, newParticle)
        numSampled += 1
        r += 1.0 / N
    end
    return sampledParticles
end

function run_belief_update (bu::DESPOTBeliefUpdateParticle,
                            problem::DESPOTProblem,
                            particles::Array{DESPOTParticle,1},
                            action::DESPOTAction,
                            obs::DESPOTObservation,
                            config::DESPOTConfig)
    newSet = Array(DESPOTParticle, 0)
    reward::Float64 = 0.
    randomNumber::Float64 = 0.

    if OS_NAME == :Linux       
        seed = Cuint[bu.beliefUpdateSeed]
    else #Windows, etc
        srand(bu.beliefUpdateSeed)
    end
    
    # Step forward all particles
    for p in particles
        if OS_NAME == :Linux
            randomNumber = ccall((:rand_r, "libc"), Int, (Ptr{Cuint},), seed) / config.randMax
        else #Windows, etc
            randomNumber = rand()
        end
        nextState, reward, nextObs = step(problem, p.state, randomNumber, action)
        obsProbability = obs_probability(problem, obs, nextState, action)

        if obsProbability > 0.
            newParticle = DESPOTParticle(nextState, p.id, p.wt * obsProbability)
            push!(newSet, newParticle)
        end
    end
    
    normalize(newSet)

    if length(newSet) == 0
        # No resulting state is consistent with the given observation, so create
        # states randomly until we have enough that are consistent.
        warn("Particle filter empty. Bootstrapping with random states")
        numSampled = 0
        while numSampled < config.nParticles
            s = random_state(problem, convert(Uint32, bu.initBeliefUpdateSeed))
            obsProbability = obs_probability(problem, obs, s, action)
            if obsProbability > 0.
                numSampled += 1
                newParticle = DESPOTParticle (s, numSampled, obsProbability)
                push!(newSet, newParticle)
            end
        end
        normalize(newSet)
        return newSet
    end

    # Remove all particles below the threshold weight
    viableParticleIndices = Array(Int64,0)
    for i in 1:length(newSet)
        if newSet[i].wt >= config.particleWtThreshold
            push!(viableParticleIndices, i)
        end
    end
    newSet = newSet[viableParticleIndices]

    if length(newSet) != 0
        normalize(newSet)
    end

    # Resample if we have < N particles or number of effective particles drops
    # below the threshold
    numEffParticles = 0
    for p in newSet
        numEffParticles += p.wt * p.wt
    end

    numEffParticles = 1./numEffParticles
    if (numEffParticles < config.nParticles * config.numEffParticleFraction) || (length(newSet) < config.nParticles)
        resampledNewSet = sampleParticles(bu, newSet, config.nParticles, config)
        newSet = resampledNewSet
    end
    return newSet
end




