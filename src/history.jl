# Encapsulates a history of actions and observations.
type History
  actions::Array{Int64,1}
  observations::Array{Int64,1}
  History()     =  new(
                    Array(Int64,0),
                    Array(Int64,0)
                    )
end

function add(history::History, action::Int64, obs::Int64)
    push!(history.actions, action)
    push!(history.observations, obs)
end

function removeLast(history::History)
    pop!(history.actions)
    pop!(history.observations)
end

function historySize(history::History)
    return length(history.actions)
end

function truncate(history::History, d::Int64)
    history.actions = history.actions[1:d]
    history.observations = history.observations[1:d]
end
