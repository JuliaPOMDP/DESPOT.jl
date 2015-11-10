# README #

This repository contains a Julia language implementation of DESPOT POMDP algorithm, designed to work with the [POMDPs.jl](https://github.com/sisl/POMDPs.jl) API.
The original (C++) version of DESPOT was developed at National University of Singapore and can be found here:

http://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.DownloadDespot

A detailed description of the algorithm can be found in this paper:

http://www.comp.nus.edu.sg/~yenan/pub/somani2013despot.pdf

The code has been tested with Julia v0.3.6.

## Installation ##

```julia
Pkg.clone("https://github.com/sisl/DESPOT.jl.git")
```

## Dependencies ##

* POMDPs
* POMDPToolbox
* Distributions

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
 or set at a later point (but before a call to *POMDPs.solve* is made) by accessing *solver.config.[parameter name]*.

|Parameter					|Type		|Default Value	|Description												|
|---------------------------|-----------|--------------:|-----------------------------------------------------------|
|search_depth				|Int64		|90				|Maximum depth of the search tree							|
|main_seed					|Uint32		|42				|The main random seed used to derive other seeds			|
|time_per_move				|Float64	|1				|CPU time allowed per move (in sec), -1 for unlimited		|
|n_particles				|Int64		|500			|Number of particles used for belief representation			|
|pruning_constant			|Float64	|0.0			|Regularization parameter									|
|eta						|Float64	|0.95			|eta*width(root) is the target uncertainty to end a trial	|
|sim_len					|Int64		|-1				|Number of moves to simulate, -1 for unlimited				|
|approximate_ubound			|Bool		|false			|If true, solver can allow initial lower bound > upper bound|
|tiny						|Float64	|1e-6			|Smallest significant difference between a pair of numbers	|
|rand_max					|Int64		|2^32-1			|Largest possible random number								|
|debug						|Uint8		|0				|Level of debug output (0-5), 0 - no output, 5 - most output|


## Instantiating the default belief updater ##
A default particle-filtering belief update type, DESPOTBeliefUpdater, is provided in the package. 

|Parameter					|Type		|Default Value	|Description												|
|---------------------------|-----------|--------------:|-----------------------------------------------------------|
|seed						|Uint32		|42				|Random seed used in belief updates							|
|n_particles				|Int64		|500			|Number of particles used for belief representation			|
|particle_weight_threshold	|Float64	|1e-20			|Smallest viable particle weight							|
|eff_particle_fraction		|Float64	|0.05			|Min. fraction of effective particles to avoid resampling	| 

Note that the solver and the belief updater values for *n_particles* should be the same (execution will be stopped
if they are different). It is also recommended to use the same *rand_max* value.

Custom belief updaters can be used as well, as long as they are based on the *DESPOTBelief* particle belief type (see *src/DESPOT.jl*).
 Please see [POMDPs.jl](https://github.com/sisl/POMDPs.jl) documentation for information on defining and using belief updaters.
 
## Solver customization ##

A DESPOT solver can be customized with user-provided upper and lower bound functions (which can also be problem-specific).

### Upper bound estimation ###

The default type of upper bound provided is a non-stochastic estimate based on value iteration (see *src/upperBound/upperBoundNonStochastic.jl*).
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

An example problem-specific lower bound type (*RockSampleParticleLB*) and the associated methods are provided for the RockSample problem
(*/src/problems/RockSample/rockSampleParticleLB.jl*). The algorithm for this lower bound estimator is based on dynamic programming.
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
So far, however, it has been tested only with the included [RockSample](https://github.com/sisl/DESPOT.jl/tree/master/src/problems/RockSample).
 
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
