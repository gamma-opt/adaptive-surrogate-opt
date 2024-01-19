# using .NNSurrogate   # To load a module from a locally defined module, a dot needs to be added before the module name like using .ModuleName.
using Surrogates
using Flux
using Random
using Statistics
using JuMP
using BSON

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
L_bounds_init = fill(-4.0, 10)
U_bounds_init = fill(-2.0, 10)
data_ST = generate_data(styblinski_tang, [L_bounds_init, U_bounds_init], 1000, SobolSample(), 0.8)

# provide the configurations
config1_ST = NN_Config([10,512,256,1], [relu, relu, identity], false, 0.1, 0.0, Flux.Optimise.Optimiser(Adam(0.01, (0.9, 0.999)), ExpDecay(0.1)), 1, 800, 1000, 0)
# config2_ST = NN_Config([10,512, 1], [relu, identity], false, 1, 0.5, Adam(0.001, (0.9, 0.999)), 5, 140, 500, 0)
# config3_ST = NN_Config([10,512,512,512,512,1], [relu, relu, relu, relu, identity], false, 0.1, 0.5, Adam(), 5, 140, 500, 0)
# configs_ST = [config1_ST]
result_ST = NN_train(data_ST, config1_ST)
NN_results(config1_ST, result_ST)

# plot_learning_curve(config1_ST, results_ST[config1_ST].err_hist)

# get the trained neural network model from the results
model_init = result_ST.model

# save/load the trained model
BSON.@save "ST_model_init.bson" model_init 
BSON.@load "ST_model_init.bson" model_init

L_bounds = vcat(Float32.(L_bounds_init), fill(Float32(-1e6), 769))
U_bounds = vcat(Float32.(U_bounds_init), fill(Float32(1e6), 769))

MILP_model = JuMP_Model(model_init, L_bounds, U_bounds)
optimize!(MILP_model)
f_hat_init = objective_value(MILP_model)
x_star_init = get_x_star(MILP_model)
f_init = styblinski_tang(Tuple(x_star_init))
gap_init = abs(f_hat_init - f_init)

#= 1st iteration =#

# generate new samples around x_star
delta = 1/2     # half of the initial interval 
L_bounds_1st = x_star_init .- 2*delta
U_bounds_1st = x_star_init
data_ST_1st = generate_data(styblinski_tang, [L_bounds_1st, U_bounds_1st], 1000, SobolSample(), 0.8)

# retrain the neural network model using the new samples, considering freezing the weights of the first few layers (c.freeze = 1)
config1_ST = NN_Config([10,512,256,1], [relu, relu, identity], false, 0.1, 0.0, Flux.Optimise.Optimiser(Adam(0.01, (0.9, 0.999)), ExpDecay(0.1)), 1, 800, 1000, 1)
result_ST = NN_train(data_ST_1st, config1_ST, trained_model = model_init)
NN_results(config1_ST, result_ST)

# get the trained neural network model from the results
model_1st = result_ST.model
BSON.@save "ST_model_1st.bson" model_1st
BSON.@load "ST_model_1st.bson" model_1st

L_bounds = vcat(Float32.(L_bounds_1st), fill(Float32(-1e6), 769))
U_bounds = vcat(Float32.(U_bounds_1st), fill(Float32(1e6), 769))

MILP_model_1st = JuMP_Model(model_1st, L_bounds, U_bounds)
optimize!(MILP_model_1st)
f_hat_1st = objective_value(MILP_model_1st)
x_star_1st = get_x_star(MILP_model_1st)
f_1st = styblinski_tang(Tuple(x_star_1st))
gap_1st = abs(f_hat_1st - f_1st)

#= 2nd iteration =#

delta = 1/2/2    # half of the previous delta
L_bounds_2nd = x_star_1st .- delta
U_bounds_2nd = x_star_1st .+ delta

data_ST_2nd = generate_data(styblinski_tang, [L_bounds_2nd, U_bounds_2nd], 1000, SobolSample(), 0.8)
config1_ST = NN_Config([10,512,256,1], [relu, relu, identity], false, 0.1, 0.0, Flux.Optimise.Optimiser(Adam(0.01, (0.9, 0.999)), ExpDecay(0.1)), 1, 800, 1000, 1)
result_ST = NN_train(data_ST_2nd, config1_ST, trained_model = model_1st)
NN_results(config1_ST, result_ST)

model_2nd = result_ST.model
BSON.@save "ST_model_2nd.bson" model_2nd
BSON.@load "ST_model_2nd.bson" model_2nd

L_bounds = vcat(Float32.(L_bounds_2nd), fill(Float32(-1e6), 769))
U_bounds = vcat(Float32.(U_bounds_2nd), fill(Float32(1e6), 769))

MILP_model_2nd = JuMP_Model(model_2nd, L_bounds, U_bounds)
optimize!(MILP_model_2nd)
f_hat_2nd = objective_value(MILP_model_2nd)
x_star = get_x_star(MILP_model_2nd)
f_2nd = styblinski_tang(Tuple(x_star))
gap_2nd = abs(f_hat_2nd - f_2nd)
