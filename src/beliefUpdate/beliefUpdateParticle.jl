import POMDPs: update
using DESPOT

type DESPOTBeliefUpdater <: POMDPs.BeliefUpdater
    pomdp::POMDP
    num_updates::Int64
    rng::DESPOTDefaultRNG
    transition_distribution::AbstractDistribution
    observation_distribution::AbstractDistribution
    state_type::DataType
    action_type::DataType
    observation_type::DataType
    seed::UInt32
    rand_max::Int64
    belief_update_seed::UInt32
    particle_weight_threshold::Float64
    eff_particle_fraction::Float64
    
    
    #pre-allocated variables (TODO: add the rest at some point)
    n_particles::Int64
    next_state::POMDPs.State
    observation::POMDPs.Observation
    new_particle::DESPOTParticle
    n_sampled::Int64
    obs_probability::Float64
    
    #default constructor
    function DESPOTBeliefUpdater(pomdp::POMDP;
                                 seed::UInt32 = convert(UInt32, 42),
                                 rand_max::Int64 = 2147483647,
                                 n_particles = 500,
                                 particle_weight_threshold::Float64 = 1e-20,
                                 eff_particle_fraction::Float64 = 0.05)
        this = new()
        this.pomdp = pomdp
        this.num_updates = 0                               
        this.belief_update_seed = seed $ (n_particles + 1)       
        this.rng = DESPOTDefaultRNG(this.belief_update_seed, rand_max)
        this.transition_distribution  = POMDPs.create_transition_distribution(pomdp)
        this.observation_distribution = POMDPs.create_observation_distribution(pomdp)
        this.state_type = typeof(POMDPs.create_state(pomdp))
        this.action_type = typeof(POMDPs.create_action(pomdp))
        this.observation_type = typeof(POMDPs.create_observation(pomdp))
        this.rand_max = rand_max
        this.particle_weight_threshold = particle_weight_threshold
        this.eff_particle_fraction = eff_particle_fraction
        this.n_particles = n_particles
        
        # init preallocated variables
        this.next_state = POMDPs.create_state(pomdp)
        this.observation = POMDPs.create_observation(pomdp)
        this.new_particle = DESPOTParticle{typeof(this.next_state)}(this.next_state, 1, 1) #placeholder
        this.n_sampled = 0
        this.obs_probability = -1.0
        return this
    end
end

# Special create_belief version for DESPOTBeliefUpdater
function create_belief(bu::DESPOTBeliefUpdater)
    particles = Array(DESPOTParticle{bu.state_type}, bu.n_particles) 
    history = History{bu.action_type, bu.observation_type}()
    belief = DESPOTBelief{bu.state_type}(particles, history)
    return belief
end

function get_belief_update_seed(bu::DESPOTBeliefUpdater)
  return bu.seed $ (bu.n_particles + 1)
end

function reset_belief(bu::DESPOTBeliefUpdater)
    bu.num_updates = 0
end

#TODO: figure out why particles::Vector{Particle} does not work
function normalize!(particles::Vector) 
  prob_sum = 0.0
  for p in particles
    prob_sum += p.weight
  end
  for p in particles
    p.weight /= prob_sum
  end
end

function update(bu::DESPOTBeliefUpdater,
                current_belief::DESPOTBelief,
                action::POMDPs.Action,
                obs::POMDPs.Observation,
                updated_belief::DESPOTBelief = create_belief(pomdp))
                
    if bu.n_particles != length(current_belief.particles)
        err("belief size mismatch: belief updater - $(bu.n_particles) particles, belief - $(length(current_belief.particles))")  
    end
    updated_belief.particles = []

    #reset RNG
    bu.rng = DESPOTDefaultRNG(bu.belief_update_seed, bu.rand_max)

    for p in current_belief.particles
        rand_num = rand!(bu.rng) #TODO: preallocate for speed
        rng = DESPOTRandomNumber(rand_num)
        
        POMDPs.transition(bu.pomdp, p.state, action, bu.transition_distribution)
        POMDPs.rand!(rng, bu.next_state, bu.transition_distribution) # update state to next state

        #get observation distribution for (s,a,s') tuple
        POMDPs.observation(bu.pomdp, p.state, action, bu.next_state, bu.observation_distribution)
        
        bu.obs_probability = pdf(bu.observation_distribution, obs)
        
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
        while bu.n_sampled < bu.n_particles
#            random_state(pomdp, convert(UInt32, bu.belief_update_seed), s)
            s = create_state(bu.pomdp) #TODO: this can be done better
            rand!(resample_rng, s, states(bu.pomdp)) #generate a random state
            bu.obs_probability = pdf(bu.observation_distribution, bu.observation)
            if bu.obs_probability > 0.0
                bu.n_sampled += 1
                bu.new_particle = DESPOTParticle(s, bu.obs_probability)
                push!(updated_belief.particles, bu.new_particle)
            end
        end
        normalize!(updated_belief.particles)
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
        resampled_set = Array(DESPOTParticle{bu.state_type}, bu.n_particles)
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




