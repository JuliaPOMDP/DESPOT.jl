mutable struct History{A,O}
    actions::Vector{A}
    observations::Vector{O}
    History{A,O}() where {A,O} = new(Array{A}(0), Array{O}(0))
end

function add{A,O}(history::History{A,O},
            action::A,
            obs::O)

    push!(history.actions, action)
    push!(history.observations, obs)
end

function remove_last{A,O}(history::History{A,O})
    pop!(history.actions)
    pop!(history.observations)
end

function history_size{A,O}(history::History{A,O})
    return length(history.actions)
end

function truncate{A,O}(history::History{A,O}, d::Int64)
    history.actions = history.actions[1:d]
    history.observations = history.observations[1:d]
end
