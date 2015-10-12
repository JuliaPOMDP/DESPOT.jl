# Encapsulates a history of actions and observations.
# type History
#   actions::Array{Int64,1}
#   observations::Array{Int64,1}
#   History()     =  new(
#                     Array(Int64,0),
#                     Array(Int64,0)
#                     )
# end

#TODO: change to parametric
type History
  actions::Array{Any,1}
  observations::Array{Any,1}
  History()     =  new(
                    Array(Any,0),
                    Array(Any,0)
                    )
end

#function add(history::History, action::Int64, obs::Int64)
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
