import POMDPs: update, initialize_belief, updater
using DESPOT

mutable struct DESPOTBeliefUpdater{S,A,O} <: POMDPs.Updater
    pomdp::POMDP
    num_updates::Int64
    rng::AbstractRNG
    seed::UInt32
    rand_max::Int64
    belief_update_seed::UInt32
    particle_weight_threshold::Float64
    eff_particle_fraction::Float64

    #pre-allocated variables (TODO: add the rest at some point)
    n_particles::Int64
    next_state::S
    observation::O
    new_particle::DESPOTParticle{S}
    n_sampled::Int64
    obs_probability::Float64

    #default constructor
    function DESPOTBeliefUpdater{S,A,O}(pomdp::POMDP{S,A,O};
                                 seed::UInt32 = convert(UInt32, 42),
                                 rand_max::Int64 = 2147483647,
                                 n_particles = 500,
                                 particle_weight_threshold::Float64 = 1e-20,
                                 eff_particle_fraction::Float64 = 0.05,
                                 next_state::S = S(),
                                 observation::O = O(),
                                 rng::AbstractRNG=DESPOTDefaultRNG(convert(UInt32, seed ⊻ (n_particles+1)), rand_max)
                                ) where {S,A,O}
        this = new()
        this.pomdp = pomdp
        this.num_updates = 0
        this.belief_update_seed = seed ⊻ (n_particles + 1)
        this.rng = rng
        this.rand_max = rand_max
        this.particle_weight_threshold = particle_weight_threshold
        this.eff_particle_fraction = eff_particle_fraction
        this.n_particles = n_particles

        # init preallocated variables
        this.next_state = next_state
        this.observation = observation
        this.new_particle = DESPOTParticle{S}(this.next_state, 1, 1) #placeholder
        this.n_sampled = 0
        this.obs_probability = -1.0
        return this
    end
end

# Special create_belief version for DESPOTBeliefUpdater
create_belief{S,A,O}(bu::DESPOTBeliefUpdater{S,A,O}) =
    DESPOTBelief{S,A,O}(Array{DESPOTParticle{S}}(bu.n_particles), History{A,O}())

get_belief_update_seed(bu::DESPOTBeliefUpdater) = bu.seed ⊻ (bu.n_particles + 1)

reset_belief(bu::DESPOTBeliefUpdater) = bu.num_updates = 0

function initialize_belief{S,A,O}(bu::DESPOTBeliefUpdater{S,A,O},
                  state_distribution::ParticleDistribution{S})

    new_belief = create_belief(bu)
    n_particles = length(state_distribution.particles)

    # convert to DESPOTParticle type
    pool = Array{DESPOTParticle{S}}(n_particles)

    for i in 1:n_particles
        pool[i] = DESPOTParticle{S}(state_distribution.particles[i].state,
                                 i, # id
                                 state_distribution.particles[i].weight)
    end

    DESPOT.sample_particles!(new_belief.particles,
                             pool,
                             bu.n_particles,
                             bu.belief_update_seed,
                             bu.rand_max)

    #shuffle!(new_belief.particles) #TODO: uncomment if higher randomness is required
    return new_belief
end

#XXX this could probably be combined with the method above or made more efficient without the sample_particles!
function initialize_belief{S}(bu::DESPOTBeliefUpdater{S}, b)
    new_belief = create_belief(bu)

    pool = Array{DESPOTParticle{S}}(bu.n_particles)
    w = 1.0/bu.n_particles
    for i in 1:bu.n_particles
        pool[i] = DESPOTParticle{S}(rand(bu.rng, b), i, w)
    end

    DESPOT.sample_particles!(new_belief.particles,
                             pool,
                             bu.n_particles,
                             bu.belief_update_seed,
                             bu.rand_max)

    return new_belief
end

updater{S,A,O}(p::DESPOTPolicy{S,A,O}) = DESPOTBeliefUpdater{S,A,O}(p.pomdp,
                                                                    n_particles=p.solver.config.n_particles,
                                                                    next_state=p.solver.next_state,
                                                                    observation=p.solver.curr_obs,
                                                                    rng=p.solver.rng
                                                                   )

function normalize!{S}(particles::Vector{DESPOTParticle{S}})
    prob_sum = 0.0
    for p in particles
        prob_sum += p.weight
    end
    for p in particles
        p.weight /= prob_sum
    end
end

function update{S,A,O}(bu::DESPOTBeliefUpdater{S,A,O},
                current_belief::DESPOTBelief{S},
                action::A,
                obs::O)

    updated_belief::DESPOTBelief{S} = create_belief(bu)
    random_number = Array{Float64}(1)

    if bu.n_particles != length(current_belief.particles)
        err("belief size mismatch: belief updater - $(bu.n_particles) particles, belief - $(length(current_belief.particles))")
    end
    updated_belief.particles = []

    #reset RNG
    if isa(bu.rng, DESPOTDefaultRNG)
        bu.rng = DESPOTDefaultRNG(bu.belief_update_seed, bu.rand_max)
    end

    for p in current_belief.particles
        rand!(bu.rng, random_number)

        if isa(bu.rng, DESPOTDefaultRNG)
            rng = DESPOTRandomNumber(random_number[1])
        else
            rng = bu.rng
        end

        bu.next_state = POMDPs.generate_s(bu.pomdp, p.state, action, rng)

        #get observation distribution for (s,a,s') tuple
        od = POMDPs.observation(bu.pomdp, p.state, action, bu.next_state)

        bu.obs_probability = pdf(od, obs)

        if bu.obs_probability > 0.0
            bu.new_particle = DESPOTParticle(bu.next_state, p.id, p.weight * bu.obs_probability)
            push!(updated_belief.particles, bu.new_particle)
        end
    end

    normalize!(updated_belief.particles)

    if length(updated_belief.particles) == 0
        # No resulting state is consistent with the given observation, so create
        # states randomly until we have enough that are consistent.
        warn("Particle filter empty. Bootstrapping with random states")
        bu.n_sampled = 0
        resample_rng = DESPOTDefaultRNG(bu.belief_update_seed, bu.rand_max)
        particle_number::Int64 = 0
        while bu.n_sampled < bu.n_particles
            #TODO: see if this can be done better
            #Pick a random particle from the current belief state as the initial state
            rand!(resample_rng, random_number)
            particle_number = ceil(bu.n_particles * random_number[1])

            next_state = POMDPs.rand(resample_rng, states(bu.pomdp)) #generate a random state
            od = POMDPs.observation(bu.pomdp,
                               current_belief.particles[particle_number].state,
                               action,
                               next_state)
            bu.obs_probability = pdf(od, bu.observation)
            if bu.obs_probability > 0.0
                bu.n_sampled += 1
                bu.new_particle = DESPOTParticle(next_state, bu.obs_probability)
                push!(updated_belief.particles, bu.new_particle)
            end
        end
        normalize!(updated_belief.particles)
        return updated_belief.particles
    end

    # Remove all particles below the threshold weight
    viable_particle_indices = Array{Int64}(0)
    for i in 1:length(updated_belief.particles)
        if updated_belief.particles[i].weight >= bu.particle_weight_threshold
            push!(viable_particle_indices, i)
        end
    end
    updated_belief.particles = updated_belief.particles[viable_particle_indices]

    if length(updated_belief.particles) != 0
        normalize!(updated_belief.particles)
    end

    # Resample if we have < N particles or number of effective particles drops
    # below the threshold
    num_eff_particles = 0
    for p in updated_belief.particles
        num_eff_particles += p.weight^2
    end

    num_eff_particles = 1./num_eff_particles
    if (num_eff_particles < bu.n_particles * bu.eff_particle_fraction) ||
        (length(updated_belief.particles) < bu.n_particles)
        resampled_set = Array{DESPOTParticle{S}}(bu.n_particles)
        sample_particles!(resampled_set,
                          updated_belief.particles,
                          bu.n_particles,
                          bu.belief_update_seed,
                          bu.rand_max)
        updated_belief.particles = resampled_set
    end

    # Finally, update history
    add(updated_belief.history, action, obs)

    return updated_belief
end
