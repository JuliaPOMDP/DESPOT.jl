# This type encapsulates the streams of random numbers used in state
# transitions during simulations. It also provides random-number seeds
# for different components of the system.

type RandomStreams
    num_streams::Int64
    len_streams::Int64
    streams::Array{Float64,2}     # each particle is associated with a single stream of numbers
    seed::UInt32

  # default constructor
  function RandomStreams(num_streams::Int64,
                len_streams::Int64,
                seed::UInt32)
          this = new() 
          
          this.num_streams = num_streams
          this.len_streams = len_streams
          this.streams = Array(Float64, num_streams, len_streams)
          this.seed = seed
          
          return this
    end
end

get_stream_seed(streams::RandomStreams, streamId::UInt32) =
    streams.seed $ streamId # bitwise XOR

get_world_seed(streams::RandomStreams) =
    streams.seed $ streams.num_streams

get_model_seed(streams::RandomStreams) =
    streams.seed $ (streams.num_streams + 2)

function fill_random_streams!(empty_streams::RandomStreams, rand_max::Int64)
    # Populate random streams
    if is_linux()
        ccall((:srand, "libc"), Void, (Cuint,), 1)
        for i in 1:empty_streams.num_streams
            seed = Cuint[get_stream_seed(empty_streams, convert(UInt32, i-1))]
            ccall( (:rand_r, "libc"), Int, (Ptr{Cuint},), seed)
            for j = 1:empty_streams.len_streams
                empty_streams.streams[i,j] = ccall((:rand_r, "libc"), Int, (Ptr{Cuint},), seed) / rand_max
            end
        end
    else #Windows, etc
        for i in 1:empty_streams.num_streams
            seed = get_stream_seed(empty_streams, convert(UInt32, i-1))
            srand(seed)
            empty_streams.streams[i,:] = Base.rand(convert(Int64, empty_streams.len_streams))
        end
    end  
end

function set_rng_state!(rng::DESPOTRandomNumber, rs::RandomStreams, scenario::Int, depth::Int)
    rng.number = rs.streams[scenario, depth]
end

create_rng(::RandomStreams) = DESPOTRandomNumber(-1)


### MersenneTwister Streams ###

type MersenneStreamArray
    rng::AbstractRNG
    streams::Vector{Vector{MersenneTwister}}
end
MersenneStreamArray(rng::AbstractRNG) = MersenneStreamArray(rng, Vector{MersenneTwister}[])

function set_rng_state!(rng::MersenneTwister, sa::MersenneStreamArray, scenario::Int, depth::Int)
    while length(sa.streams) < scenario
        push!(sa.streams, MersenneTwister[])
    end
    stream = sa.streams[scenario]
    while length(stream) < depth
        push!(stream, MersenneTwister(rand(sa.rng, UInt32))) # maybe just advance a large number of steps and copy, rather than seed?
    end
    copy!(rng, stream[depth])
end

fill_random_streams!(streams::MersenneStreamArray, rand_max::Int64) = nothing

create_rng(::MersenneStreamArray) = copy(Base.GLOBAL_RNG)
