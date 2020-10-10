# In this file lie the function for Gillepsie algorithm
"""
$(TYPEDEF)
"""
struct Gillepsie <: AbstractAlg end
export Gillepsie
"""
    function give_birth(a::Agent,t,p::Dict)
Used for Gillepsie setting
"""
function give_birth(mum_idx::Int,w::World)
    new_a = copy(w[mum_idx])
    increment_x!(new_a,space(w),parameters(w),time(w))
    return new_a
end

function updateBirthEvent!(w::World,::Gillepsie,mum_idx::Int)
    # updating competition only the two columns corresponding to agent idx
    @unpack d,b = parameters(w)
    offspring = give_birth(mum_idx,w)
    x_offspring = get_x(offspring)
    _agents = agents(w)
    for a in _agents
        a.d += d(get_x(a),x_offspring,w.t)
    end
    # Now updating new agent
    offspring.d = sum(d.(get_x.(_agents),Ref(x_offspring),w.t)) #- d(x_offspring,x_offspring)
    offspring.b = b(x_offspring,w.t)
    addAgent!(w,offspring)
end

function updateDeathEvent!(world::World,::Gillepsie,i_event::Int)
    @unpack d = parameters(world)
    x_death = get_x(world[i_event])
    # updating death rate only the two columns corresponding to agent idx
    removeAgent!(world,i_event)
    for a in agents(world)
        a.d -= d(get_x(a),x_death,world.t)
    end
end

"""
    update_rates_std!(world,p::Dict,t::Float64)
This standard updates takes
    - competition kernels of the form α(x,y) and
    - carrying capacity of the form K(x)
"""
function  update_rates!(w::World,::Gillepsie)
    @unpack b,d = parameters(w)
    _agents = agents(w)
    traits = get_x.(_agents)
    # traits = get_xhist.(world)
    n = size(w)
    D = zeros(n)
    # Here you should do a shared array to compute in parallel
    for i in 1:(n-1)
        for j in i+1:n
            C = d(traits[i],traits[j],w.t)
            D[i] += C
            D[j] += C
        end
    end
    # Here we can do  it in parallel as well
    for (i,a) in enumerate(_agents)
        a.d = D[i]
        a.b = b(traits[i],w.t)
    end
end

"""
    function updateWorld_G!(world,t,p)
Updating rule for gillepsie setting.
Returning dt drawn from an exponential distribution with parameter the total rates of events.
 # Args
 t is for now not used but might be used for dynamic landscape
"""
function updateWorld!(w::World{A,S,T},g::G) where {A,S,T,G <: Gillepsie}
    # total update world
    alive = agents(w)
    events_weights = ProbabilityWeights(vcat(get_d.(alive),get_b.(alive)))
    # Total rate of events
    ∑ = sum(events_weights)
    dt = - log(rand(T))/∑
    update_clock!(w,dt)
    if dt > 0.
        i_event = sample(events_weights)
        # This is ok since size is a multiple of 2
        n = size(w)
        if i_event <= n
            # DEATH EVENT
            # In this case i_event is also the index of the individual to die in the world_alive
            updateDeathEvent!(w,g,i_event)
        else
            # birth event
            # i_event - n is also the index of the individual to give birth in the world_alive
            updateBirthEvent!(w,g,i_event-n)
        end
        return dt
    else
        return -1
    end
end

"""
    clean_world(world::Array{T}) where {T <: Agent}
Get rid of missing value in `world`
"""
#TODO : put some type specs here
clean_world(world) = collect(skipmissing(world))
