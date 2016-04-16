[![Build Status](https://travis-ci.com/JuliaPOMDP/DESPOT.jl.svg?branch=master)](https://travis-ci.com/JuliaPOMDP/DESPOT.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaPOMDP/DESPOT.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaPOMDP/DESPOT.jl?branch=master)

# README #

This repository contains a Julia language implementation of DESPOT POMDP algorithm, designed to work with the [POMDPs.jl](https://github.com/sisl/POMDPs.jl) API.
The original (C++) version of DESPOT was developed at National University of Singapore and can be found here:

http://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.DownloadDespot

A detailed description of the algorithm can be found in this paper:

http://www.comp.nus.edu.sg/~yenan/pub/somani2013despot.pdf

The code has been tested with Julia v0.4.2.

## Installation ##

```julia
Pkg.clone("https://github.com/sisl/DESPOT.jl")
```

## Dependencies ##

* POMDPs
* POMDPToolbox
* Distributions

## Data types ##

The following DESPOT-specific types are likely to be of interest to problem and application developers:

|Type				|Supertype		|Comments						|
|-------------------------------|-----------------------|-------------------------------------------------------|
|DESPOTSolver			|POMDPs.Solver		|The main solver type 											|
|DESPOTUpperBound		|Any			|An abstract type for defining types and functions for computing an upper bound			| 
|DESPOTLowerBound		|Any			|An abstract type for defining types and functions for computing a lower bound			|
|DESPOTPolicy			|POMDPs.Policy		|A custom policy type											|
|DESPOTParticle			|Any			|A custom particle type used by the solver and the default belief updater				|
|DESPOTBelief			|POMDPs.Belief		|A custom belief type containing both a particle-based belief and a solver history log		|
|DESPOTConfig			|Any			|A set of DESPOT configuration parameters								|
|DESPOTDefaultRNG		|POMDPs.AbstractRNG	|The default multi-platform RNG type that can be used to advance the state of the simulation 	| 

When defining problem-specific POMDPs.State, POMDPs.Action, and POMDPs.Observation subtypes, the problem developer needs to make sure that *hash()* functions and *==* operators for these subtypes are defined as well, as they are required by the solver. Problem-specific state, action, and observation spaces must be defined as iterable types, either by using existing iterable containers (such as arrays) or by defining *start()*, *next()*, and *finish()* functions for them. For more on this subject, please see [POMDPs.jl documentation](https://github.com/sisl/POMDPs.jl) and Julia documentation on [iteration](http://docs.julialang.org/en/latest/stdlib/collections/#iteration).

## Instantiating a DESPOT solver ##

The following example illustrates instantiation of a DESPOT solver
```julia
solver = DESPOTSolver(pomdp,			# reference to the problem model
					  current_belief, 	# reference to the current belief structure
					  lb = custom_lb, 	# reference to the optional custom lower bound
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

A DESPOT solver can be customized with user-provided upper and lower bound functions (which can also be problem-specific).

### Upper bound estimation ###

The default type of upper bound provided is a non-stochastic estimate based on value iteration, defined in [upperBoundNonStochastic.jl](src/upperBound/upperBoundNonStochastic.jl).
 To add a custom upper bound algorithm, define a custom type as a subtype of *DESPOTUpperBound*, e.g.:

```julia
type MyUpperBound <: DESPOTUpperBound
```
then define functions to initialize and compute the upper bound with the following signatures:

```julia
function upper_bound(::MyUpperBound, 			# upper bound variable
					 ::POMDP,					# problem model
					 ::Vector{DESPOTParticle}, 	# belief of interest represented via particles
					 ::DESPOTConfig)			# solver configuration parameters

function init_upper_bound(::MyUpperBound,
						  ::POMDP,
						  ::DESPOTConfig)
```

Instantiate an object of type *MyUpperBound*, e.g.:
```julia
my_ub = MyUpperBound(pomdp)
```
Then pass it to the DESPOT solver as a keyword argument:
```julia
solver = DESPOTSolver(pomdp, 			# reference to the problem model
                      current_belief, 	# reference to the current belief structure
                      ub = my_ub) 		# reference to the optional custom upper bound
```

### Lower bound estimation ###

An example problem-specific lower bound type and the associated methods are provided for the RockSample problem in [rockSampleParticleLB.jl](/src/problems/RockSample/rockSampleParticleLB.jl). The algorithm for this lower bound estimator is based on dynamic programming.
 Similarly to upper bounds, to add a custom upper bound algorithm, define a custom type as a subtype of *DESPOTLowerBound*, e.g.:

```julia
type MyLowerBound <: DESPOTLowerBound
```
then define functions to initialize and compute the upper bound with the following signatures:

```julia
function upper_bound(::MyLowerBound, 			# upper bound variable
					 ::POMDP,					# problem model
					 ::Vector{DESPOTParticle}, 	# belief of interest represented via particles
					 ::DESPOTConfig)			# solver configuration parameters

function init_upper_bound(::MyLowerBound,
						  ::POMDP,
						  ::DESPOTConfig)
```

Instantiate an object of type *MyLowerBound*, e.g.:
```julia
my_lb = MyLowerBound(pomdp)
```
Then pass it to the DESPOT solver as a keyword argument:
```julia
solver = DESPOTSolver(pomdp,		   	# reference to the problem model
                      current_belief, 	# reference to the current belief structure
                      ub = my_ub, 		# reference to the optional custom upper bound
                      lb = my_lb) 		# reference to the optional custom lower bound
```

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
Runtime = 6.30 sec  

## Bugs ##

Please feel free to file bug reports and I will try to address them as soon as I am able.
 Feature requests will be considered as well, but only as time allows.
