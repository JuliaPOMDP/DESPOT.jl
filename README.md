# README #

This repository contains a Julia language implementation of DESPOT POMDP algorithm, The original (C++) version was developed at National University of Singapore and can be found here:

http://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.DownloadDespot

A detailed description of the algorithm can be found in this paper:

http://www.comp.nus.edu.sg/~yenan/pub/somani2013despot.pdf

The code has been tested with Julia v0.3.6.

## Dependencies ##

* Distributions

## Running DESPOT on test problems ##

### Rock Sample ###
RockSample is the only problem currently available. To run once, use main([grid size],[number of rocks]). To run in the batch mode main([grid size],[number of rocks], [number of repetitions]).

### Example output for RockSample ###

Upon successful execution, you should see the following output:

**Single run of main(4,4)**:

Number of steps = 15

Discounted return = 26.21

Undiscounted return = 40.00

Average number of trials per move = 120.27

Runtime = 5.90 sec

**Batch run main(4,4,5)**:

================= Batch Averages =================

Number of steps = 15

Discounted return = 26.21

Undiscounted return = 40.00

Average number of trials per move = 118.16

Runtime = 6.28 sec

## Code ##

### Core code ###

- despot.jl - the entry point for running DESPOT. Contains initialization and configuration code, as well as main() methods for execution. Currently the default (and only) problem is RockSample, but other problems will be added in the future. This file should be considered a template for your application code - consider making a copy of it and modifying it for your own needs.

- solver.jl - the core DESPOT algorithm

- config.jl - data type defining algorithm parameters, including discount factor, search depth, time per move, etc. The actual values of these parameters are set in despot.jl. The purpose of each parameter is described in config.jl.

- randomStreams.jl - data type and methods for managing streams of random numbers (needed by DESPOT for creating scenario sets)

- types.jl - other data types needed by DESPOT (e.g. Particle and StateProbability)

- utils.jl - utility methods needed by the solver (mostly for quantifying uncertainty)

- vnode.jl - VNode data type and methods for representing belief nodes

- qnode.jl - QNode data type and methods for representing action nodes

- history.jl - data type and methods for storing rollout history

- world.jl - data type and methods for managing the state of problem

### Upper Bound ###

The directory contains code for computing the initial upper bound estimate. Currently the only type of upper bound provided is a non-stochastic estimate based on value iteration.

To add custom upper bound code, define a custom type as a subtype of UpperBound, e.g.:

*MyUpperBound <: UpperBound*

and add the file with your upper bound code to the upperBound directory. The interface to your upper bound methods will be implementation-specific.

### Lower Bound ###

This is a placeholder directory for generalized lower bound estimation code. In the current implementation of RockSample problem-specific lower bound methods are used. 

### Belief Update ###

This directory contains code for performing belief updates. Currently only a particle-filtering belief update method is provided, however custom methods can be added.

### Problems ###

This directory contains test problems. As mentioned above, only RockSample is provided at the moment. Additional problems can be added by creating a custom type and 

*MyProblem <: Problem*

the following methods need to be implemented:

*nextState::Int64, reward::Float64, observation::Int64 = step(problem::MyProblem, state::Int64, randNum::Float64, action::Int64)*

The above is a standard POMDP transition function, with only randNum being a DESPOT-specific addition (please see the 2013 DESPOT paper listed above for details)

*ans::Bool = isTerminal(problem::MyProblem, s::Int64)*

returns true if the state is terminal and false otherwise. Optional functions for printing state, observations, etc can be implemented as well. Please see RockSample code for examples. 

## Bugs ##

Please feel free to file bug reports and I will try to address them as soon as I am able. Feature requests will be considered, but only as time allows.