# using .NNSurrogate   # To load a module from a locally defined module, a dot needs to be added before the module name like using .ModuleName.
using Surrogates
using Flux
using Random
using Statistics

include("../src/NNSurrogate.jl")
include("../src/NNJuMP.jl")

"""
- Styblinski-Tang function (10 variables)
"""

# set random seed
Random.seed!(1)

# define the function we are going to build surrogate for
styblinski_tang(x::Tuple) = 0.5 * sum([xi^4 - 16*xi^2 + 5*xi for xi in x])

# sampling
data_ST = generate_data(styblinski_tang, [fill(-4.0, 10), fill(-2.0, 10)], 1000, SobolSample(), 0.8)

# provide the configurations
config1_ST = NN_Config([10,512,256,1], [relu, relu, identity], false, 0.1, 0.5, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, 800, 1000)
# config2_ST = NN_Config([10,512, 1], [relu, identity], false, 1, 0.5, Adam(0.001, (0.9, 0.999)), 5, 140, 500)
# config3_ST = NN_Config([10,512,512,512,512,1], [relu, relu, relu, relu, identity], false, 0.1, 0.5, Adam(), 5, 140, 500)
configs_ST = [config1_ST]

# trian the nerual net
results_ST = NN_compare(data_ST, configs_ST)
for (configs, results) in results_ST
    println("Layers: $(configs.layer), Activation: $(configs.af), Epochs: $(configs.epochs), Train Error: $(results.train_err), Test Error: $(results.test_err)")
end

plot_learning_curve(config1_ST, results_ST[config1_ST].err_hist)

NN_model = results_ST[config1_ST].model
y_pred = NN_model(data_ST.x_test)
y_act = data_ST.y_test

L_bounds = vcat(fill(Float32(-4.0), 10), fill(Float32(-1e6), 769))
U_bounds = vcat(fill(Float32(-2.0), 10), fill(Float32(1e6), 769))

MILP_model = JuMP_Model(NN_model, L_bounds, U_bounds)
optimize!(MILP_model)
println(objective_value(MILP_model))
println(value.(MILP_model[:x][0,:]))
solution_summary(MILP_model)
