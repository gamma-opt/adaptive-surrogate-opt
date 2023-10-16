# using .NNSurrogate
using Surrogates
using Flux
using Random
using Statistics

include("../src/NNSurrogate.jl")
include("../src/NNJuMP.jl")

"""
- Goldstein-Price function (2 variables)
"""

# set random seed
Random.seed!(1)

# define the function we are going to build surrogate for
goldstein_price(x::Tuple) = (1+(x[1]+x[2]+1)^2*(19-14*x[1]+3*x[1]^2-14*x[2]+6*x[1]*x[2]+3*x[2]^2)) * 
    (30+(2*x[1]-3*x[2])^2*(18-32*x[1]+12*x[1]^2+48*x[2]-36*x[1]*x[2]+27*x[2]^2))

# sampling
data_GP = generate_data(goldstein_price, [Float32[-0.5, -1.5], Float32[0.5, -0.5]], 1000, SobolSample(), 0.8)

# provide the configurations
config1_GP = NN_Config([2,1024,1], [relu, identity], false, 0.01, 0.4, Flux.Optimise.Optimiser(Adam(1, (0.89, 0.899)), ExpDecay(1, 0.08, 1000, 1e-4)), 1, 800, 5000)
# config2_GP = NN_Config([2,512,1], [relu, identity], false, 0.001, 0.4, Adam(),  1, 800, 1000)
# config3_GP = NN_Config([2,512,1], [relu, identity], 0.001, 0.4, Adam(), 1000)
configs_GP = [config1_GP]

# trian the nerual net
results_GP = NN_compare(data_GP, configs_GP)
for (configs, results) in results_GP
    println("Layers: $(configs.layer), Activation: $(configs.af), Epochs: $(configs.epochs), Train Error: $(results.train_err), Test Error: $(results.test_err)")
end

NN_model = results_GP[config1_GP].model
y_pred = NN_model(data_GP.x_test)
y_act = data_GP.y_test

plot_learning_curve(config1_GP, results_GP[config1_GP].err_hist)

# set corresponding bounds to the input layer, and arbitrary large big-M in the other layers
L_bounds = vcat(Float32[-0.5, -1.5], fill(Float32(-1e6), 1025))
U_bounds = vcat(Float32[0.5, -0.5], fill(Float32(1e6), 1025))

# convert the trained nerual net to a JuMP model
MILP_model = JuMP_Model(NN_model, L_bounds, U_bounds)

optimize!(MILP_model)
println(objective_value(MILP_model))
println(value.(MILP_model[:x][0,:]))
solution_summary(MILP_model)