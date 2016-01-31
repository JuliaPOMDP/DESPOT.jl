
type History{A,O}
    actions::Vector{A}#Array{A,1}
    observations::Vector{O}#Array{O,1}

    History() = new(Array(A, 0), Array(O, 0))
#     function History()
#         this = new()
#         this.actions = Array(ActionType, 0)
#         this.observations = Array(ObservationType, 0)
#         this.ActionType = ActionType
#         this.ObservationType = ObservationType
#         return this
#     end
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
