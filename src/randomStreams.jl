
# This class encapsulates the streams of random numbers used in state
# transitions during simulations. It also provides random-number seeds
# for different components of the system.


type RandomStreams
    numStreams::Uint32
    lenStreams::Uint32
    streams::Array{Float64,2}     # each particle is associated with a single stream of numbers
    seed::Uint32
    worldSeed::Uint32
    beliefUpdateSeed::Uint32
    modelSeed::Uint32

  # default constructor
  function RandomStreams(numStreams::Uint32,
                lenStreams::Uint32,
                seed::Uint32)
          this = new() 
          
          this.numStreams = numStreams
          this.lenStreams = lenStreams
          this.streams = Array(Float64, numStreams, lenStreams)
          this.seed = seed
          this.worldSeed = seed $ numStreams
          this.beliefUpdateSeed = seed $ (numStreams + 1)
          this.modelSeed = seed $ (numStreams + 2)
          
          return this
    end
end

function getStreamSeed(streams::RandomStreams, streamId::Uint32)
  return streams.seed $ streamId # bitwise XOR
end

# function getWorldSeed(streams::RandomStreams)
#   return streams.seed $ streams.numStreams
# end
# 
# function getBeliefUpdateSeed(streams::RandomStreams)
#   return streams.seed $ (streams.numStreams + 1)
# end
# 
# function getModelSeed(streams::RandomStreams)
#   return streams.seed $ (streams.numStreams + 2)
# end

function fillRandomStreams(emptyStreams::RandomStreams, randMax::Int64)
    # Populate random streams
    if OS_NAME == :Linux
        ccall((:srand, "libc"), Void, (Cuint,), 1)
        for i in 1:emptyStreams.numStreams
            seed = Cuint[getStreamSeed(emptyStreams, convert(Uint32, i-1))]
            ccall( (:rand_r, "libc"), Int, (Ptr{Cuint},), seed)
            for j = 1:emptyStreams.lenStreams
                emptyStreams.streams[i,j] = ccall((:rand_r, "libc"), Int, (Ptr{Cuint},), seed) / randMax
            end
        end
    else #Windows, etc
        for i in 1:emptyStreams.numStreams
            seed  = getStreamSeed(emptyStreams, convert(Uint32, i-1))
            srand(seed)
            emptyStreams.streams[i,:] = rand(convert(Int64, emptyStreams.lenStreams))
        end
    end  
end
