#using DESPOT
import DESPOT

include("rockSample.jl")
include("../../upperBound/upperBoundNonStochastic.jl")
include("../../beliefUpdate/beliefUpdateParticle.jl")

function main(gridSize::Int64 = 4, numRocks::Int64 = 4)

    # create a DESPOTPomdp object 
    problem = RockSample(gridSize, numRocks) 
    pomdp = DESPOTPomdp (problem)
    init_problem(pomdp)

    # Here is how you can adjust the default DESPOT parameters, if they were not passed
    # through the optional arguments of the DESPOTPomdp constructor above (if desired).
    
    # control computational resource use either by limiting timePerMove
    # or by limiting the number of trials per move (or both). Setting either
    # to 0 or a negative number disables that limit.
    
    pomdp.config.searchDepth = 90
    pomdp.config.discount = 0.95
    pomdp.config.rootSeed = 42
    pomdp.config.timePerMove = 1                 # sec
    pomdp.config.nParticles = 500
    pomdp.config.pruningConstant = 0
    pomdp.config.eta = 0.95
    pomdp.config.simLen = -1
    pomdp.config.approximateUBound = false
    pomdp.config.particleWtThreshold = 1e-20
    pomdp.config.numEffParticleFraction = 0.05
    pomdp.config.tiny = 1e-6
    pomdp.config.maxTrials = -1
    pomdp.config.randMax = 2147483647
    pomdp.config.debug = 0
    
    UpperBoundNonStochastic(pomdp)
    belief = initial_belief(problem)  # create initial belief from the problem's initial belief
    solver = DESPOTSolver(pomdp)
    policy = solve (solver, pomdp)
    
    simStep = 0

    tic() # start the clock
    while (!is_finished(solver, pomdp) && 
        (pomdp.config.simLen == -1 || simStep < pomdp.config.simLen))
        a = action(policy, belief)
        #println("In testRockSample: $(methods(step))")
        obs, reward = execute_action(pomdp, a)
#        obs, reward = step(pomdp, a)
        update_belief!(belief, pomdp, a, obs)
        simStep += 1
    end
    runTime = toq() # stop the clock
    
    @printf("Number of steps = %d\n", simStep)
    @printf("Discounted return = %.2f\n", discounted_return(pomdp))
    @printf("Undiscounted return = %.2f\n", undiscounted_return(pomdp))
    @printf("Runtime = %.2f sec\n", runTime)
end
