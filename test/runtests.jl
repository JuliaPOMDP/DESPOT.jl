using Base.Test

# uncomment below to include compatibility tests
include("pomdps_compatibility_tests.jl")

if is_windows()
    error("This test is only valid on Linux and OS X platforms at this time") 
end

# Test on a simple RockSample problem
include("../src/problems/RockSample/runRockSample.jl")
include("test_with_other_particle_filter.jl")
include("test_mersenne.jl")

# Common problem parameters
n_particles             = 500 # for both solver and belief updater
main_seed               = 42
discount                = 0.95
search_depth            = 90
time_per_move           = -1. #default: 1
pruning_constant        = 0.
eta                     = 0.95
sim_len                 = -1
max_trials              = 100 #default: -1
approximate_ubound      = false
debug                   = 0

# Standard batch (3 runs) RockSample(4,4) test
grid_size               = 4
num_rocks               = 4

# The return values below are batch averages 
sim_steps, undiscounted_return, discounted_return, run_time =
                main(
                    grid_size = grid_size,
                    num_rocks = num_rocks,
                    n_particles = n_particles,
                    main_seed = main_seed,
                    discount = discount,
                    search_depth = search_depth,
                    time_per_move = time_per_move,
                    pruning_constant = pruning_constant, 
                    eta = eta,
                    sim_len = sim_len,
                    max_trials = max_trials,
                    approximate_ubound = approximate_ubound,
                    debug = debug,
                    n_reps = 3
                    )

#println(sim_steps, undiscounted_return, discounted_return, run_time)
if is_linux()
    @test               sim_steps == 11
    @test_approx_eq_eps discounted_return 12.62 1e-2
    @test               undiscounted_return == 20.00
end
# osx tests
if is_apple()
    @test               sim_steps == 11
    @test_approx_eq_eps discounted_return 12.97 1e-2
    @test               undiscounted_return == 20.00
end

println("DESPOT/RockSample(4,4) batch test status: PASSED")



# Non-standard RockSample(5,6) test 
# (non-standard means that the initial state space is generated programmatically)

grid_size               = 5
num_rocks               = 6

sim_steps, undiscounted_return, discounted_return, run_time =
            execute(
                    grid_size = grid_size,
                    num_rocks = num_rocks,
                    n_particles = n_particles,
                    main_seed = main_seed,
                    discount = discount,
                    search_depth = search_depth,
                    time_per_move = time_per_move,
                    pruning_constant = pruning_constant, 
                    eta = eta,
                    sim_len = sim_len,
                    max_trials = max_trials,
                    approximate_ubound = approximate_ubound,
                    debug = debug
                    )

println(sim_steps, undiscounted_return, discounted_return, run_time)
if is_linux()
    @test               sim_steps == 21
    @test_approx_eq_eps discounted_return 28.97 1e-2
    @test               undiscounted_return == 50.00
end
# osx tests
if is_apple()
    @test               sim_steps == 20
    @test_approx_eq_eps discounted_return 20.80 1e-2
    @test               undiscounted_return == 40.00
end

println("DESPOT/RockSample(5,6) test status: PASSED")


# Standard RockSample(7,8) test
grid_size               = 7
num_rocks               = 8


sim_steps, undiscounted_return, discounted_return, run_time =
            execute(
                    grid_size = grid_size,
                    num_rocks = num_rocks,
                    n_particles = n_particles,
                    main_seed = main_seed,
                    discount = discount,
                    search_depth = search_depth,
                    time_per_move = time_per_move,
                    pruning_constant = pruning_constant, 
                    eta = eta,
                    sim_len = sim_len,
                    max_trials = max_trials,
                    approximate_ubound = approximate_ubound,
                    debug = debug
                    )

#println(sim_steps, undiscounted_return, discounted_return, run_time)


if is_linux()
    @test               sim_steps == 32
    @test_approx_eq_eps discounted_return 24.82 1e-2
    @test               undiscounted_return == 50.00
end
# osx tests
if is_apple()
    @test               sim_steps == 30
    @test_approx_eq_eps discounted_return 16.98 1e-2
    @test               undiscounted_return == 40.00
end

println("DESPOT/RockSample(7,8) test status: PASSED")


# Standard RockSample(11,11) test
grid_size               = 11
num_rocks               = 11

sim_steps, undiscounted_return, discounted_return, run_time =
            execute(
                    grid_size = grid_size,
                    num_rocks = num_rocks,
                    n_particles = n_particles,
                    main_seed = main_seed,
                    discount = discount,
                    search_depth = search_depth,
                    time_per_move = time_per_move,
                    pruning_constant = pruning_constant, 
                    eta = eta,
                    sim_len = sim_len,
                    max_trials = max_trials,
                    approximate_ubound = approximate_ubound,
                    debug = debug
                    )

#println(sim_steps, undiscounted_return, discounted_return, run_time)
if is_linux()
    @test               sim_steps == 50
    @test_approx_eq_eps discounted_return 21.24 1e-2
    @test               undiscounted_return == 60.00
end
# osx tests
if is_apple()
    @test               sim_steps == 45
    @test_approx_eq_eps discounted_return 13.5 1e-2
    @test               undiscounted_return == 40.00
end

println("DESPOT/RockSample(11,11) test status: PASSED")
