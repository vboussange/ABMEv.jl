# In this file lie the function for Gillepsie algorithm


"""
    give_birth(a::Agent,p)
Used for Gillepsie setting
"""
function give_birth(a::Agent,t,p::Dict)
    new_a = copy(a)
    increment_x!(new_a,t,p)
    return new_a
end

function update_afterbirth_std!(world,idx_offspring,t,p::Dict) where T
    α = p["alpha"];K=p["K"];
    x_offspring = get_x(world[idx_offspring])
    for a in skipmissing(world)
        a.d += α(get_x(a),x_offspring)
        a.b = K(get_x(a),t)
    end
    # rectifying new agent competition experienced
    world[idx_offspring].d = sum(α.(get_x.(skipmissing(world)),Ref(x_offspring))) - α(x_offspring,x_offspring)
end

function update_afterdeath_std!(world,x_death,t,p::Dict) where T
    α = p["alpha"];K=p["K"];
    for a in skipmissing(world)
        a.d -= α(get_x(a),x_death)
        a.b = K(get_x(a),t)
    end
end

"""
    function updateWorld_G!(world,t,p)
Updating rule for gillepsie setting.
Returning dt drawn from an exponential distribution with parameter the total rates of events.
 # Args
 t is for now not used but might be used for dynamic landscape
"""
function updateWorld_G!(world,p,update_rates!,t)
    # total update world
    world_alive = skipmissing(world)
    idx_world = collect(eachindex(world_alive))
    # Total rate of events
    U = sum(get_d.(world_alive) .+ get_b.(world_alive))
    dt = - log(rand(Float64))/U
    if dt > 0.
        events_weights = ProbabilityWeights(vcat(get_d.(world_alive),get_b.(world_alive)))
        i_event = sample(events_weights)
        # This is ok since length is a multiple of 2
        N = length(world) - count(ismissing,world)
        if i_event <= N
            # DEATH EVENT
            # In this case i_event is also the index of the individual to die in the world_alive
            idx_offspring = idx_world[i_event]
            x_death = get_x(world[idx_offspring])
            update_afterdeath_std!(world,x_death,t+dt,p)
            world[idx_offspring] = missing
        else
            # BIRTH EVENT
            idx_offspring = findfirst(ismissing,world)
            # i_event - N is also the index of the individual to give birth in the world_alive
            mum = world[idx_world[i_event-N]]
            world[idx_offspring] = give_birth(mum,t+dt,p)
            update_afterbirth_std!(world,idx_offspring,t+dt,p)
        end
        return dt
    else
        return -1.
    end
end

"""
    clean_world(world::Array{T}) where {T <: Agent}
Get rid of missing value in `world`
"""
#TODO : put some type specs here
clean_world(world) = collect(skipmissing(world))
