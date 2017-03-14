# DESPOT.jl
[![Build Status](https://travis-ci.org/JuliaPOMDP/DESPOT.jl.svg?branch=master)](https://travis-ci.org/JuliaPOMDP/DESPOT.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaPOMDP/DESPOT.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaPOMDP/DESPOT.jl?branch=master)

This repository contains a Julia language implementation of DESPOT POMDP algorithm (http://www.comp.nus.edu.sg/~yenan/pub/somani2013despot.pdf), designed to work with the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) API. 

A C++ implementation of DESPOT was developed at National University of Singapore and can be found here:

http://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.DownloadDespot

The code has been tested with Julia v0.5.0.

## Installation ##

```julia
using POMDPs
POMDPs.add("DESPOT")
```

## Dependencies ##

* [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl)
* [POMDPToolbox.jl](https://github.com/JuliaPOMDP/POMDPToolbox.jl)
* [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)

## Data types ##

The following DESPOT-specific types are likely to be of interest to problem and application developers:

|Type				|Supertype		|Comments						|
|-------------------------------|-----------------------|-------------------------------------------------------|
|DESPOTSolver			|POMDPs.Solver		|The main solver type 											|
|DESPOTPolicy			|POMDPs.Policy		|A custom policy type											|
|DESPOTParticle			|Any			|A custom particle type used by the solver and the default belief updater				|
|DESPOTBelief			|Any			|A custom belief type containing both a particle-based belief and a solver history log		|
|DESPOTConfig			|Any			|A set of DESPOT configuration parameters								|
|DESPOTDefaultRNG		|POMDPs.AbstractRNG	|The default multi-platform RNG type that can be used to advance the state of the simulation 	| 

When defining problem-specific state, action, and observation types, the problem developer needs to make sure that *hash()* functions and *==* operators for these types are defined as well, as they are required by the solver. Problem-specific state, action, and observation spaces must be defined as iterable types, either by using existing iterable containers (such as arrays) or by defining *start()*, *next()*, and *finish()* functions for them. For more on this subject, please see [POMDPs.jl documentation](https://github.com/sisl/POMDPs.jl) and Julia documentation on [iteration](http://docs.julialang.org/en/latest/stdlib/collections/#iteration).

## Instantiating a DESPOT solver ##

The following example illustrates instantiation of a DESPOT solver
```julia
solver = DESPOTSolver(lb = custom_lb, 	# reference to the optional custom lower bound
					  ub = custom_ub) 	# reference to the optional custom upper bound
```

Information on how to construct custom upper and lower bound estimators is provided in section Customization.
Additional solver parameters (listed below) can either also be passed as keyword arguments during the solver construction
 or set at a later point (but before a call to *POMDPs.solve* is made) by accessing *solver.config.[parameter]*.

|Parameter					|Type		|Default Value	|Description												|
|---------------------------|-----------|--------------:|-----------------------------------------------------------|
|search_depth				|Int64		|90				|Maximum depth of the search tree							|
|main_seed					|UInt32		|42				|The main random seed used to derive other seeds			|
|time_per_move				|Float64	|1				|CPU time allowed per move (in sec), -1 for unlimited		|
|n_particles				|Int64		|500			|Number of particles used for belief representation			|
|pruning_constant			|Float64	|0.0			|Regularization parameter									|
|eta						|Float64	|0.95			|eta*width(root) is the target uncertainty to end a trial	|
|sim_len					|Int64		|-1				|Number of moves to simulate, -1 for unlimited				|
|approximate_ubound			|Bool		|false			|If true, solver can allow initial lower bound > upper bound|
|tiny						|Float64	|1e-6			|Smallest significant difference between a pair of numbers	|
|rand_max					|Int64		|2^32-1			|Largest possible random number								|
|debug						|UInt8		|0				|Level of debug output (0-5), 0 - no output, 5 - most output|
|random_streams             |           |`RandomStreams`|Source of random numbers. `RandomStreams` is designed to reproduce the behavior in the DESPOT paper, `MersenneStreamArray` is designed to be more widely compatible. See [pomdps_compatibility_tests.jl](test/pomdps_compatibility_tests.jl) for examples.


## Instantiating the default belief updater ##
A default particle-filtering belief update type, DESPOTBeliefUpdater, is provided in the package. 

|Parameter					|Type		|Default Value	|Description												|
|---------------------------|-----------|--------------:|-----------------------------------------------------------|
|seed						|UInt32		|42				|Random seed used in belief updates							|
|n_particles				|Int64		|500			|Number of particles used for belief representation			|
|particle_weight_threshold	|Float64	|1e-20			|Smallest viable particle weight							|
|eff_particle_fraction		|Float64	|0.05			|Min. fraction of effective particles to avoid resampling	| 

Note that the solver and the belief updater values for *n_particles* should be the same (execution will be stopped
if they are different). It is also recommended to use the same *rand_max* value.

Custom belief updaters can be used as well, as long as they are based on the *DESPOTBelief* particle belief type (see [DESPOT.jl](src/DESPOT.jl)).
 Please see [POMDPs.jl](https://github.com/sisl/POMDPs.jl) documentation for information on defining and using belief updaters.
 
## Solver customization ##

### Bounds

A DESPOT solver can be customized with user-provided bounds (which can also be problem-specific).

To define bounds, a user should define a custom type (e.g. `MyUpperBound`) and implement a function with the following signature

```julia
DESPOT.bounds{S}(::MyUpperBound, ::POMDP, ::Vector{DESPOTParticle{S}}, ::DESPOTConfig)
```
that returns a tuple containing the lower bound and upper bound values. Some examples can be found in [pomdps_compatibility_tests.jl](test/pomdps_compatibility_tests.jl), [upperBoundNonStochastic.jl](src/upperBound/upperBoundNonStochastic.jl), and [rockSampleParticleLB.jl](/src/problems/RockSample/rockSampleParticleLB.jl).



## Running DESPOT on test problems ##

DESPOT.jl should be compatible with test problems in [POMDPModels.jl](https://github.com/sisl/POMDPModels.jl).
So far, however, it has been tested only with the included [RockSample](src/problems/RockSample).
 
### Rock Sample ###
To run a RockSample problem in REPL, for example, do the following:

```julia
include("runRockSample.jl")
main([grid size],[number of rocks])
```

Running main() without arguments will execute a simple RockSample example with 4 rocks on a grid size of 4.

### Example output for RockSample ###

Upon successful execution, you should see output of the following form:

********** EXECUTION SUMMARY **********  
Number of steps = 11  
Undiscounted return = 20.00  
Discounted return = 12.62  
Runtime = 2.45 sec

## Tree Visualization ##

An example of how to visualize the search tree can be found in [test_visualization.jl](test/test_visualization.jl).

## Performance ##
Benchmarking results for DESPOT (on RockSample) can be found in [perflog.md](https://github.com/JuliaPOMDP/DESPOT.jl/test/perflog.md). As more problems are tested with DESPOT, their benchmarks will be included as well. 

## Bugs ##

Please feel free to file bug reports and I will try to address them as soon as I am able.
 Feature requests will be considered as well, but only as time allows.
