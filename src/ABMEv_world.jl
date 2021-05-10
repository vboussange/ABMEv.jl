# this used to be the worldalive

# TODO: do a constructor that ensures the parameters numerics are of the same type as the agents
mutable struct World{A<:AbstractAgent, S<:AbstractSpacesTuple,T<:Number}
    agents::Vector{A}
    space::S
    p::Dict
    t::T
end

#constructor
function World(w::Vector{A},s::S,p::Dict,t::T=0.) where {A<:AbstractAgent,S<:AbstractSpacesTuple,T}
    # if typeof(p["D"]) != eltype(skipmissing(w)[1])
    #     throw(ArgumentError("Diffusion coefficient does not match with underlying space\n `D::Tuple`"))
    # end
    if typeof(first(w)) !== A
        throw(ArgumentError("eltype(w) should be equal to typeof(first(w)), which is not the case.\n
                            Try to generate the array of Agents in an other way"))
    end
    for _m in mu
        if typeof(m) !<: Float
            throw(ArgumentError("elements of mu should be of type AbstractFloat\n
                                to decide if mutations occur from a uniform probability law"))
        end
    end
    length(mu) == ndims(s) ? nothing : "Dimension of parameter mu should correspond to dimension of underlying space"
    length(D) == ndims(s) ? nothing : "Dimension of parameter D should correspond to dimension of underlying space"

    World{A,S,T}(w,s,p,t)
end

# this throws an iterators of agents in the world
agents(world::World) = world.agents
parameters(world::World) = world.p
time(w::World) = w.t
space(w::World) = w.space
maxsize(w::World) = w.p["NMax"]
# this throws indices that are occupied by agents
# this throws agents of an abstract array of size size(world)
import Base:size,getindex
Base.size(world::World) = length(world.agents)
Base.copy(w::W) where {W<:World} = W(copy.(w.agents),w.space,w.p,copy(w.t))
## Accessors
"""
$(SIGNATURES)
Get x of world without geotrait.
"""
Base.getindex(w::World,i) = w.agents[i]

function Base.show(io::IO, w::World{A,S,T}) where {A,S,T}
     println(io, "World with agents of type", A)
 end

addAgent!(w::World,a::AbstractAgent) = begin
    push!(w.agents,a)
end
removeAgent!(w::World,i::Int) = begin
    deleteat!(w.agents,i)
end

update_clock!(w::World{A,S,T},dt) where {A,S,T} = begin
    w.t = convert(T,sum(w.t + dt))
    return nothing
end


"""
$(SIGNATURES)
"""
get_geo(w::World) = map(a-> get_geo(a,time(w)), agents(w))

"""
$(SIGNATURES)
Returns trait of every agents of world in the form of an array which dimensions corresponds to the input.
If `trait = 0` , we return the geotrait.
!!! warning "Warning"
    Geotrait might be deprecated in the future.
"""
function get_x(w::World,trait)
    if !(trait == 0)
        if ndims(space(w)[trait]) > 1
            return hcat(collect.(getindex.(agents(w),trait))...)'
        else
            return collect(getindex.(agents(w),trait))
        end
    else
        return collect(get_geo(w))
    end
end

"""
$(SIGNATURES)
Returns every traits of every agents of `world` in the form **of a one dimensional array** (in contrast to `get_x`).
If `geotrait=true` the geotrait is also added to the set of trait, in the last column.
If you do not want to specify `t` (only useful for geotrait), it is also possible to use `get_xarray(world::Array{T,1}) where {T <: Agent}`.
!!! warning "Warning"
    It does not work with subspace where ndims(subspace) > 1.
"""
function get_xarray(world::World,geotrait::Bool=false)
    xarray = get_x(world,Colon())
    if geotrait
        xarray = hcat(xarray, get_geo.(agents(world),world.t))
    end
    return xarray
end
@deprecate get_xarray(world,geotrait=false) get_x(world,Colon())

"""
    function give_birth(mum_idx::Int,w::World)
Copies agent within index `mum_idx`, and increment it by dx.
Return new agent (offspring).
"""
function give_birth(mum_idx::Int,w::World)
    new_a = copyxt(w[mum_idx])
    increment_x!(new_a,space(w),parameters(w),time(w))
    return new_a
end