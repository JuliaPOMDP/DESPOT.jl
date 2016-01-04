
type History{ActionType, ObservationType}
    actions::Array{ActionType, 1}
    observations::Array{ObservationType, 1}

    History() = new(Array(ActionType, 0), Array(ObservationType, 0))
#     function History()
#         this = new()
#         this.actions = Array(ActionType, 0)
#         this.observations = Array(ObservationType, 0)
#         this.ActionType = ActionType
#         this.ObservationType = ObservationType
#         return this
#     end
end

function add(history::History,
            action::POMDPs.Action,
            obs::POMDPs.Observation)
    
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
