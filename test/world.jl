using LightGraphs
using Test
using Revise,EvoId
using UnPack
myspace = (GraphSpace(SimpleGraph(10,10)),RealSpace{1,Float64}())
myagents = [Agent(myspace,ancestors=true,rates=true) for i in 1:10]
d(X,Y,t) = gaussian(X[1],Y[1],1)
b(X,Y,t) = gaussian(X[1],0,1)
D = Any[Float32(1),Float64(1.)]
mu = [1.,1.]
NMax = 100
p = Dict{String,Any}();@pack! p = d,b,D,mu,NMax



@test eltype(myagents) <: AbstractAgentM
@test typeof(myagents) <: Vector{A} where {A<:AbstractAgentM}

w = EvoId.World(myagents,myspace,p)
@test typeof(w.p["D"][1]) == Float32
@test size(w) ≈ 10
newa = give_birth(1,w)
addAgent!(w,newa)
@test size(w) ≈ 11
removeAgent!(w,11)
@test size(w) ≈ 10
@test isnothing(update_clock!(w,.1))

@test typeof(w[1]) <: AbstractAgent
@test size(get_x(w,:)) == (10,2)

##############  Testing world with Gillepsie
@test !isnothing(updateWorld!(w,Gillepsie(),b,d))
