using ABMEv, Test, JLD2,Random

@testset "ABMEv" begin
    include("gillepsie.jl")
    # include("wrightfisher.jl")
    include("metrics.jl")
    include("metrics_hamming.jl")
    include("utilstest.jl")
    include("simulation.jl")
    include("space_agent.jl")
    include("world.jl")
    include("utils.jl")
end
