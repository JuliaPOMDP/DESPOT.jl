
type History{TA,TO}
  actions::Array{TA,1}
  observations::Array{TO,1}
  History() =  new(Array(TA,0),Array(TO,0))
end

#TODO: Change to POMDPs.jl abstract data types
function add(history::History, action::Any, obs::Any)
    push!(history.actions, action)
    push!(history.observations, obs)
end

function remove_last(history::History)
    pop!(history.actions)
    pop!(history.observations)
end

function history_size(history::History)
    return length(history.actions)
end

function truncate(history::History, d::Int64)
    history.actions = history.actions[1:d]
    history.observations = history.observations[1:d]
end
