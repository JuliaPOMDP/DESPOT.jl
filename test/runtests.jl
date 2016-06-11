using Base.Test

@windows_only error("This test is only valid on Linux and OS X platforms at this time") 

# Test on a simple RockSample problem
include("../src/problems/RockSample/runRockSample.jl")

grid_size               = 4
num_rocks               = 4
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
@test               sim_steps == 11
@linux_only @test_approx_eq_eps discounted_return 12.62 1e-2
@linux_only @test               undiscounted_return == 20.00
# osx tests
@osx_only @test_approx_eq_eps discounted_return 12.97 1e-2
@osx_only @test               undiscounted_return == 20.00

println("DESPOT/RockSample test status: PASSED")
