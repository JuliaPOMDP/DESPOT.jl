
function fringe_upper_bound(pomdp::RockSample, state::RockSampleState)
    if isterminal(pomdp, state)
        return 0.0
    end

    rock_set = rock_set_of(pomdp, state)
    n_good = 0
    while rock_set != 0
        n_good += rock_set & 1
        rock_set >>>= 1
    end

    # Assume a good rock is sampled at each step and an exit is made in the last
    if pomdp.discount < 1
        return 10. * (1 - (pomdp.discount^(n_good+1))) / (1 - pomdp.discount)
    else
        return 10. * (n_good + 1)
    end
end
