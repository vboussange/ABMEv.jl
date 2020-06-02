# for now we consider that competition is local within an array

"""
    update_rates_std!(world,p::Dict,t::Float64)
This standard updates takes
    - competition kernels of the form α(x,y) and
    - carrying capacity of the form K(x)
"""
function  update_rates_std!(world,t::Float64,p::Dict)
    α = p["alpha"];K=p["K"];
    traits = get_x.(world)
    # traits = get_xhist.(world)
    N = length(traits)
    D = zeros(N)
    # Here you should do a shared array to compute in parallel
    for i in 1:(N-1)
        for j in i+1:N
            C = α(traits[i],traits[j])
            D[i] += C
            D[j] += C
        end
    end
    # Here we can do  it in parallel as well
    for (i,a) in enumerate(world)
        a.d = D[i]
        a.b = K(traits[i],t)
    end
end

function  update_rates_graph!(world,C,t::Float64,p::Dict)
    for e in edges(p["g"])
        # agents on e
        aidx_e = findall(a -> get_x(a,1)==e,world)
        na = length(aidx_e)
        for i in aidx_e
            world[i].d = na^2
            world[i].b = 1
        end
    end
end

function  update_rates_2D!(world,C,p::Dict,t::Float64)
    # Competition matrix
    traits = get_x.(world)
    # traits = get_xhist.(world)
    N = length(traits)
    # C = SharedArray{Float64}((N,N))
    # Here you should do a shared array to compute in parallel
    for i in 1:(N-1)
        C[i,i] = 1.
        for j in i+1:N
            # be careful, for xhist what is return is an array hence no need of []
            C[i,j] = α(traits[i],traits[j],p["n_alpha"],p["sigma_a"])
            C[j,i] = C[i,j]
        end
    end
    C[N,N] = 1.
    # Here we can do  it in parallel as well
    for (i,a) in enumerate(world)
        # we only update death rate
        # a.d = sum(C[i,:]) / K(traits[i][2:2],p["K0"],p["n_K"],p["sigma_K"], μ = p["a"]*traits[i][1])
        a.d = sum(C[i,:])
        # /!| not ideal to assign at every time step the birth rate that is constant
        a.b = KK(traits[i][1:1],
                    p["K0"],
                    p["n_K"],
                    p["sigma_K"][1:1],
                    split_merge_move(t),
                    -split_merge_move(t))/K(traits[i][2:2],
                    p["K0"],
                    p["n_K"],
                    p["sigma_K"][2:2])
    end
end

function  update_rates_grad2D!(world,C,p::Dict,t::Float64)
    # Competition matrix
    traits = get_x.(world)
    # traits = get_xhist.(world)
    N = length(traits)
    # C = SharedArray{Float64}((N,N))
    # Here you should do a shared array to compute in parallel
    for i in 1:(N-1)
        C[i,i] = 1.
        for j in i+1:N
            # be careful, for xhist what is return is an array hence no need of []
            C[i,j] = α(traits[i],traits[j],p["n_alpha"],p["sigma_a"])
            C[j,i] = C[i,j]
        end
    end
    C[N,N] = 1.
    # Here we can do  it in parallel as well
    for (i,a) in enumerate(world)
        # we only update death rate
        a.d = sum(C[i,:])
        a.b = K(traits[i][2:2],p["K0"],p["n_K"],p["sigma_K"], μ = p["a"]*traits[i][1])
    end
end

function  update_rates_mountain!(world,C,p::Dict,t::Float64)
    # Competition matrix
    traits = get_x.(world)
    # traits = get_xhist.(world)
    N = length(traits)
    # C = SharedArray{Float64}((N,N))
    # Here you should do a shared array to compute in parallel
    for i in 1:(N-1)
        C[i,i] = 1.
        for j in i+1:N
            # be careful, for xhist what is return is an array hence no need of []
            C[i,j] = α(traits[i],traits[j],p["n_alpha"],p["sigma_a"])
            C[j,i] = C[i,j]
        end
    end
    C[N,N] = 1.
    # Here we can do  it in parallel as well
    for (i,a) in enumerate(world)
        # we only update death rate
        a.d = sum(C[i,:])
        a.b = KK(traits[i][1:1],
                p["K0"],
                p["n_K"],
                p["sigma_K"][1:1],
                split_move(t),
                -split_move(t))/K(traits[i][2:2],
                p["K0"],
                p["n_K"],
                p["sigma_K"][2:2],
                μ=(tin(traits[i][1],-1.,0.)*(traits[i][1]+1) - tin(traits[i][1],0.,1.)*(traits[i][1]-1) ) * split_move(t))
    end
end

function  update_rates_std_split!(world,C,p::Dict,t::Float64)
    # Competition matrix
    traits = get_x.(world)
    # traits = get_xhist.(world)
    N = length(traits)
    # C = SharedArray{Float64}((N,N))
    # Here you should do a shared array to compute in parallel
    for i in 1:(N-1)
        C[i,i] = 1.
        for j in i+1:N
            C[i,j] = α(traits[i],traits[j],p["n_alpha"],p["sigma_a"])
            C[j,i] = C[i,j]
        end
    end
    C[N,N] = 1.
    # Here we can do  it in parallel as well
    for (i,a) in enumerate(world)
        # we only update death rate
        # this could be imptrove since \alpha is symmetric, by using a symmetric matrix
        # a.d = sum(C[i,:]) / KK(traits[i][:,end],p["K0"],p["n_K"],p["sigma_K"],split_merge_move(t),-split_merge_move(t))
        a.d = sum(C[i,:])
        # /!| not ideal to assign at every time step the birth rate that is constant
        a.b =  KK(traits[i][:,end],p["K0"],p["n_K"],p["sigma_K"],split_move(t),-split_move(t))
    end
end
"""
    function runWorld_store_WF(p,world0;mode="std")
Wright Fisher process. Returns an array worldall with every agents.
"""
function runWorld_store_WF(p,world0;mode="std")
    worldall = repeat(world0,inner = (1,length(1:Int(p["tend"]))))
    N=length(world0);
    newworld = copy.(world0)
    if mode == "std"
        update_rates! = update_rates_std!
    elseif mode == "2D"
        update_rates! = update_rates_2D!
    elseif mode == "grad2D"
        update_rates! = update_rates_grad2D!
    elseif mode == "mountain"
        update_rates! = update_rates_mountain!
    elseif mode == "split"
        update_rates! = update_rates_std_split!
    elseif mode == "graph"
        update_rates! = update_rates_graph!
    else
        error("Mode $mode is not recognized")
    end
    for i in 1:(Int(p["tend"])-1)
        # we have to take views, otherwise it does not affect worldall
        world = @view worldall[:,i];newworld = @view worldall[:,i+1];
        updateWorld_WF!(world,newworld,p,update_rates!,Float64(i))
    end
    return worldall,collect(0:Int(p["tend"]-1))
end
"""
    function runWorld_store_G(p,world0)
Gillepsie process. Returns a tuple worldall,tspanarray
If specified p["dt_saving"] determines the time step to save the simulation. If not only last time step is saved

"""
function runWorld_store_G(p,world0)
    # we store the value of world every 100 changes by default
    t = .0
    i = 1;j=1;dt = 1.
    N=length(world0);
    tspanarray = zeros(1);
    dt_saving = haskey(p,"dt_saving") ? p["dt_saving"] : p["tend"] + 1.
    # Array that stores the agents
    worldall = reshape(copy.(world0),N,1);
    # We add a second row for final time step
    worldall = hcat(worldall,Array{Missing}(missing,N,1))
    # we instantiate C as the biggest size it can take
    update_rates_std!(skipmissing(world0),0.,p)
    while t<p["tend"]
        if dt < 0
            throw("We obtained negative time step dt = $dt at event $i")
        elseif count(ismissing,world0) == p["NMax"]
            @info "All individuals have died :("
            break
        elseif count(ismissing,world0) == 0
            @info "We have reached the maximum number of individuals allowed"
            break
        end
        if  t - tspanarray[end] >= dt_saving
            @info "saving world @ t = $(t)/ $(p["tend"])"
            j+=1;sw = size(worldall,2);
            # we use <= because we save at the end of the wile loop
            if sw <= j
                # we double the size of worldall
                worldall = hcat(worldall,Array{Missing}(missing,N,sw))
            end
            worldall[1:Int(N - count(ismissing,world0)),j] .= copy.(collect(skipmissing(world0)));
            push!(tspanarray,t)
        end
        dt = updateWorld_G!(world0,p,update_rates_std!,t)
        t +=  dt
        i += 1
    end
    # Saving last time step
    j+=1
    worldall[1:Int(N - count(ismissing,world0)),j] .= copy.(collect(skipmissing(world0)));
    push!(tspanarray,t)
    @info "simulation stopped at t=$(tspanarray[end]), after $(i) generations"
    return worldall[:,1:j],tspanarray
end
