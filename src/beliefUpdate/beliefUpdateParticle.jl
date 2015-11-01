import POMDPs: update
using DESPOT

type DESPOTBeliefUpdater <: POMDPs.BeliefUpdater
    pomdp::POMDP
    num_updates::Int64
    rng::DESPOTDefaultRNG
    transition_distribution::AbstractDistribution
    observation_distribution::AbstractDistribution
    state_type::DataType
    observation_type::DataType
    seed::Uint32
    rand_max::Int64
    belief_update_seed::Uint32
    particle_weight_threshold::Float64
    eff_particle_fraction::Float64
    
    
    #pre-allocated variables (TODO: add the rest at some point)
    n_particles::Int64
    next_state::Any
    observation::Any
    new_particle::Particle
    n_sampled::Int64
    obs_probability::Float64
    
    #default constructor
    function DESPOTBeliefUpdater(pomdp::POMDP;
                                 seed::Uint32 = convert(Uint32, 42),
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
        this.observation_type = typeof(POMDPs.create_observation(pomdp))
        this.rand_max = rand_max
        this.particle_weight_threshold = particle_weight_threshold
        this.eff_particle_fraction = eff_particle_fraction
        this.n_particles = n_particles
        
        # init preallocated variables
        this.next_state = POMDPs.create_state(pomdp)
        this.observation = POMDPs.create_observation(pomdp)
        this.new_particle = Particle{typeof(this.next_state)}(this.next_state, 1)
        this.n_sampled = 0
        this.obs_probability = -1.0
        return this
    end
end

# Special create_belief version for DESPOTBeliefUpdater
function create_belief(bu::DESPOTBeliefUpdater)
    particles = Array(Particle{bu.state_type},bu.n_particles) 
    history = History() #TODO: change to parametric
    belief = DESPOTBelief{bu.state_type}(particles, history)
    #return DESPOTBelief{RockSampleState}()
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
                action::Any,
                obs::Any,
                updated_belief::DESPOTBelief = create_belief(pomdp))
                
#     println("num current particles 1: $(length(current_belief.particles))")
    if bu.n_particles != length(current_belief.particles)
        err("belief size mismatch: belief updater - $(bu.n_particles) particles, belief - $(length(current_belief.particles))")  
    end
    updated_belief.particles = []

    #reset RNG
    bu.rng = DESPOTDefaultRNG(bu.belief_update_seed, bu.rand_max)

#     #TODO: is this needed here?
#     if OS_NAME == :Linux
#         seed = Cuint[bu.belief_update_seed]
#     else #Windows, etc
#         srand(bu.belief_update_seed)
#     end
    
    
    
#     println("num current particles 2: $(length(current_belief.particles))")
    #println("in update, current: $(current_belief.particles[10:15])")
# # #     # Step forward all particles
    i=1
    debug = 0
    low = 1
    high = 5
     
    println("random seed: $(bu.belief_update_seed)")
    for p in current_belief.particles
        rand_num = rand!(bu.rng) #TODO: preallocate for speed
        rng = DESPOTRandomNumber(rand_num)
        
        POMDPs.transition(bu.pomdp, p.state, action, bu.transition_distribution)
        bu.next_state = POMDPs.rand!(rng, bu.next_state, bu.transition_distribution) # update state to next state
#         if (p.state == bu.next_state)
#             println("States equal: $(p.state) and $(bu.next_state)")
#         end


         #get observation distribution for (a,s') tuple
         POMDPs.observation(bu.pomdp, p.state, action, bu.next_state, bu.observation_distribution)
#         bu.observation = POMDPs.rand!(rng, bu.observation, bu.observation_distribution)
        
        if (i <= high) && (i >= low)
            debug = 1
        else
            debug = 0
        end
        
        bu.obs_probability = pdf(bu.observation_distribution, obs, debug)
        
        if (i <= high) && (i >= low)
            println("random number [$i]: $rand_num")
            println("obs[$i]=$(obs), prob = $(bu.obs_probability)")
        end
        i=i+1
        
#         #TODO: remove
#         if (abs(bu.obs_probability - 0.02)>1.0e-6)
#             println(bu.obs_probability)
#         end
        
        if bu.obs_probability > 0.0
            bu.new_particle = Particle(bu.next_state, p.weight * bu.obs_probability)
            push!(updated_belief.particles, bu.new_particle)
        end
    end
    
#    println("bu: $(updated_belief.particles)")
#    println("before: $(updated_belief.particles[495:500])")
    normalize!(updated_belief.particles)
#    println("after: $(updated_belief.particles)")
    #println("bu norm.: $(updated_belief.particles[10:15])")

    if length(updated_belief.particles) == 0
        # No resulting state is consistent with the given observation, so create
        # states randomly until we have enough that are consistent.
        warn("Particle filter empty. Bootstrapping with random states")
        bu.n_sampled = 0
        while bu.n_sampled < bu.n_particles
            s = random_state(pomdp, convert(Uint32, bu.belief_update_seed))
            bu.obs_probability = pdf(bu.observation_distribution, bu.observation)
            if bu.obs_probability > 0.
                bu.n_sampled += 1
                bu.new_particle = Particle(s, bu.obs_probability)
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
        resampled_set = Array(Particle{bu.state_type}, bu.n_particles)
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




