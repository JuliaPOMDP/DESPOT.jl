import POMDPs: belief

type ParticleBeliefUpdater <: POMDPs.BeliefUpdater
    num_updates::Int64
    belief_update_seed::Int64
    rng::DESPOTDefaultRNG
    transition_distribution::AbstractDistribution
    observation_distribution::AbstractDistribution
    rand_max::Int64
    particle_weight_threshold::Float64
    eff_particle_fraction::Float64
    
    #pre-allocated variables (TODO: add the rest at some point)
    n_particles::Int64
    next_state::Any
    observation::Any
    new_particle::Particle
    num_sampled::Int64
    obs_probability::Float64
    
    #default constructor
    function ParticleBeliefUpdater(pomdp::POMDP,
                                   belief_update_seed::Uint32 = 42,
                                   rand_max::Int64 = 2147483647,
                                   particle_weight_threshold::Float64 = 1e-20,
                                   eff_particle_fraction::Float64 = 0.05)
        this = new()
        this.num_updates = 0                               # num_updates
        this.belief_update_seed = belief_update_seed       # belief_update_seed
        this.rng = DESPOTDefaultRNG(belief_update_seed, rand_max)
        this.transition_distribution  = POMDPs.create_transition_distribution(pomdp)
        this.observation_distribution = POMDPs.create_observation_distribution(pomdp)
        this.rand_max = rand_max
        this.particle_weight_threshold = particle_weight_threshold
        this.eff_particle_fraction = eff_particle_fraction
        
        # init preallocated variables
        this.n_particles = 0
        this.next_state = POMDPs.create_state(pomdp)
        this.observation = POMDPs.create_observation(pomdp)
        this.new_particle = Particle{typeof(this.next_state)}(this.next_state, 1)
        this.num_sampled = 0
        this.obs_probability = -1.0
        return this
    end
end

function reset_belief(bu::ParticleBeliefUpdater)
    bu.num_updates = 0
end

#TODO: figure out why particles::Vector{Particle} does not work
function normalize(particles::Vector) 
  prob_sum = 0.
  for p in particles
    prob_sum += p.weight
  end
  for p in particles
    p.weight /= prob_sum
  end
end

function belief(bu::ParticleBeliefUpdater,
                pomdp::POMDP,
                current_belief::ParticleBelief,
                action::Any,
                obs::Any,
                updated_belief::ParticleBelief = create_belief(pomdp))
                
    #new_set = Array(Particle, 0)
    bu.n_particles = length(current_belief.particles)
    updated_belief.particles = []

    if OS_NAME == :Linux
        seed = Cuint[bu.belief_update_seed]
    else #Windows, etc
        srand(bu.belief_update_seed)
    end
    
    #println("in update, current: $(current_belief.particles[10:15])")
    # Step forward all particles
    for p in current_belief.particles     
        POMDPs.transition(pomdp, p.state, action, bu.transition_distribution)
        bu.next_state = POMDPs.rand!(bu.rng, bu.next_state, bu.transition_distribution) # update state to next state
#         if (p.state == bu.next_state)
#             println("States equal: $(p.state) and $(bu.next_state)")
#         end
        POMDPs.observation(pomdp, bu.next_state, action, bu.observation_distribution)
        bu.observation = POMDPs.rand!(bu.rng, obs, bu.observation_distribution)
        bu.obs_probability = pdf(bu.observation_distribution, bu.observation)
        if bu.obs_probability > 0.0
            bu.new_particle = Particle(bu.next_state, p.weight * bu.obs_probability)
            push!(updated_belief.particles, bu.new_particle)
        end
    end
    
#     println("bu: $(updated_belief.particles[400:405])")
    normalize(updated_belief.particles)
    #println("bu norm.: $(updated_belief.particles[10:15])")

    if length(updated_belief.particles) == 0
        # No resulting state is consistent with the given observation, so create
        # states randomly until we have enough that are consistent.
        warn("Particle filter empty. Bootstrapping with random states")
        bu.num_sampled = 0
        while bu.num_sampled < bu.n_particles
            s = random_state(pomdp, convert(Uint32, bu.belief_update_seed))
            bu.obs_probability = pdf(bu.observation_distribution, bu.observation)
            if bu.obs_probability > 0.
                bu.num_sampled += 1
                bu.new_particle = Particle(s, bu.obs_probability)
                push!(updated_belief.particles, bu.new_particle)
            end
        end
        normalize(updated_belief.particles)
        return updated_belief.particles
    end

    # Remove all particles below the threshold weight
    viable_particle_indices = Array(Int64,0)
    for i in 1:length(updated_belief.particles)
        if updated_belief.particles[i].weight >= bu.particle_weight_threshold
            push!(viable_particle_indices, i)
        end
    end
    updated_belief.particles = updated_belief.particles[viable_particle_indices]

    if length(updated_belief.particles) != 0
        normalize(updated_belief.particles)
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
        resampled_new_set = sample_particles(updated_belief.particles,
                                             bu.n_particles,
                                             bu.belief_update_seed,
                                             bu.rand_max)
        updated_belief.particles = resampled_new_set
    end
    #println("bu end: $(updated_belief.particles[10:15])")
    return updated_belief.particles
end




