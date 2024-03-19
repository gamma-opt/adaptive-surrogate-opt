# using .NNSurrogate   # To load a module from a locally defined module, a dot needs to be added before the module name like using .ModuleName.
using Surrogates
using Flux
using Random
using Statistics
using JuMP
using BSON

include("../../src/NNSurrogate.jl")
include("../../src/NNJuMP.jl")
include("../../src/NNOptimise.jl")

"""
- Styblinski-Tang function (10 variables)
"""

# set random seed
Random.seed!(1)

# define the function we are going to build surrogate for
styblinski_tang(x::Tuple) = 0.5 * sum([xi^4 - 16*xi^2 + 5*xi for xi in x])

#--------------------------------- Initial training ---------------------------------#
# sampling
L_bounds_init = fill(-4.0, 10)
U_bounds_init = fill(-2.0, 10)
sampling_config_init = Sampling_Config(1000, L_bounds_init, U_bounds_init)

data_ST = generate_data(styblinski_tang, sampling_config_init, SobolSample(), 0.8)

# provide the configurations
config1_ST = NN_Config([10,512,256,1], [relu, relu, identity], false, 0.1, 0.1, Flux.Optimise.Optimiser(Adam(0.01, (0.9, 0.999)), ExpDecay(0.1)), 5, 640, 1000, 0)
# config2_ST = NN_Config([10,512, 1], [relu, identity], false, 1, 0.5, Adam(0.001, (0.9, 0.999)), 5, 140, 500, 0)
# config3_ST = NN_Config([10,512,512,512,512,1], [relu, relu, relu, relu, identity], false, 0.1, 0.5, Adam(), 5, 140, 500, 0)
# configs_ST = [config1_ST]
result_ST = NN_train(data_ST, config1_ST)
NN_results(config1_ST, result_ST)

# get the trained neural network model from the results
model_init = result_ST.model

# save/load the trained model
BSON.@save "ST_model_init.bson" model_init 
BSON.@load "ST_model_init.bson" model_init

L_bounds = vcat(Float32.(sampling_config_init.lb), fill(Float32(-1e6), sum(config1_ST.layer[2:end])))
U_bounds = vcat(Float32.(sampling_config_init.ub), fill(Float32(1e6), sum(config1_ST.layer[2:end])))

MILP_model = JuMP_Model(model_init, L_bounds, U_bounds)
optimize!(MILP_model)

f_hat, f_true, x_star_init, gap = solution_evaluate(MILP_model, styblinski_tang)

#--------------------------------- 1st iteration ---------------------------------#

#-----strategy 1: fixed percentage of the search space

# generate new samples around x_star
sampling_config_1st = generate_resample_config(sampling_config_init, x_star_init, 1.05, ("fixed_percentage_density", 1.0), "fixed_percentage")
data_ST_1st = generate_data(styblinski_tang, sampling_config_1st, SobolSample(), 0.8)
# generate new dataset with data_ST_1st and data_ST within the new defined sampling_config_1st lower and upper bounds

new_data_ST = filter_(data_ST, Float32.(sampling_config_1st.lb), Float32.(sampling_config_1st.ub))
data_ST_1st = combine_datasets(new_data_ST, data_ST_1st)

# retrain the neural network model using the new samples, considering freezing the weights of the first few layers (c.freeze = 1)
config1_ST = NN_Config([10,512,256,1], [relu, relu, identity], false, 0.1, 0.5, Flux.Optimise.Optimiser(Adam(0.01, (0.9, 0.999)), ExpDecay(0.1)), 1, round(Int, sampling_config_1st.n_samples*0.8), 1000, 1)
result_ST = NN_train(data_ST_1st, config1_ST, trained_model = model_init)
NN_results(config1_ST, result_ST)

# get the trained neural network model from the results
model_1st = result_ST.model
BSON.@save "ST_model_1st.bson" model_1st
BSON.@load "ST_model_1st.bson" model_1st

L_bounds = vcat(Float32.(sampling_config_1st.lb), fill(Float32(-1e6), sum(config1_ST.layer[2:end])))
U_bounds = vcat(Float32.(sampling_config_1st.ub), fill(Float32(1e6), sum(config1_ST.layer[2:end])))

MILP_model_1st = rebuild_JuMP_Model(model_1st, MILP_model, config1_ST.freeze, L_bounds, U_bounds)
warmstart_JuMP_Model(MILP_model_1st, x_star_init)
optimize!(MILP_model_1st)
f_hat_1st, f_true_1st, x_star_1st, gap_1st = solution_evaluate(MILP_model_1st, styblinski_tang)

#-----strategy 2: error based resampling

x_belows, x_aboves = find_max_errs(data_ST, model_init, x_star_init)
# generate new samples around x_star
sampling_config_1st_eb = generate_resample_config(sampling_config_init, x_star_init, 1.05, ("fixed_percentage_density", 1.0), "error_based", x_below = x_belows, x_above = x_aboves)
data_ST_1st_eb = generate_data(styblinski_tang, sampling_config_1st_eb, SobolSample(), 0.8)

config1_ST = NN_Config([10,512,256,1], [relu, relu, identity], false, 0.1, 0.5, Flux.Optimise.Optimiser(Adam(0.01, (0.9, 0.999)), ExpDecay(0.1)), 1, round(Int, sampling_config_init.n_samples*0.8), 1000, 1)
result_ST = NN_train(data_ST_1st_eb, config1_ST, trained_model = model_init)
NN_results(config1_ST, result_ST)

model_1st = result_ST.model

L_bounds = vcat(Float32.(sampling_config_1st_eb.lb), fill(Float32(-1e6), sum(config1_ST.layer[2:end])))
U_bounds = vcat(Float32.(sampling_config_1st_eb.ub), fill(Float32(1e6), sum(config1_ST.layer[2:end])))
MILP_model_1st_eb = rebuild_JuMP_Model(model_1st, MILP_model, config1_ST.freeze, L_bounds, U_bounds)
optimize!(MILP_model_1st_eb)
f_hat_1st_eb, f_true_1st_eb, x_star_1st_eb, gap_1st_eb = solution_evaluate(MILP_model_1st_eb, styblinski_tang)

#-----strategy 3: sagmented error based resampling
plots_array = plot_segmented_errs(data_ST, model_init, x_star_init, 50)
# Combine all the individual plots into one composite plot
plot(plots_array..., layout=(10, 1), size=(500, 200 * 10))

x_belows, x_aboves = find_max_segmented_errs(data_ST, model_init, x_star_init, 20)
# generate new samples around x_star
sampling_config_1st_seb = generate_resample_config(sampling_config_init, x_star_init, 1.05, ("fixed_percentage_density", 1.0), "segmented_error", x_below = x_belows, x_above = x_aboves)

data_ST_1st_seb = generate_data(styblinski_tang, sampling_config_1st_seb, SobolSample(), 0.8)
config1_ST = NN_Config([10,512,256,1], [relu, relu, identity], false, 0.1, 0.5, Flux.Optimise.Optimiser(Adam(0.01, (0.9, 0.999)), ExpDecay(0.1)), 1, round(Int, sampling_config_1st_seb.n_samples*0.8), 1000, 1)
result_ST = NN_train(data_ST_1st_seb, config1_ST, trained_model = model_init)
NN_results(config1_ST, result_ST)

model_1st_seb = result_ST.model

L_bounds = vcat(Float32.(sampling_config_1st_seb.lb), fill(Float32(-1e6), sum(config1_ST.layer[2:end])))
U_bounds = vcat(Float32.(sampling_config_1st_seb.ub), fill(Float32(1e6), sum(config1_ST.layer[2:end])))

MILP_model_1st_seb = rebuild_JuMP_Model(model_1st_seb, MILP_model, config1_ST.freeze, L_bounds, U_bounds)
warmstart_JuMP_Model(MILP_model_1st_seb, x_star_init)
optimize!(MILP_model_1st_seb)
f_hat_1st_seb, f_true_1st_seb, x_star_1st_seb, gap_1st_seb = solution_evaluate(MILP_model_1st_seb, styblinski_tang)

#--------------------------------- 2nd iteration ---------------------------------#

#-----strategy 1: fixed percentage of the search space

# generate new samples around x_star
sampling_config_2nd = generate_resample_config(sampling_config_1st, x_star_1st, 1.05, ("fixed_percentage_density", 1.0), "fixed_percentage")
data_ST_2nd = generate_data(styblinski_tang, sampling_config_2nd, SobolSample(), 0.8)

config1_ST = NN_Config([10,512,256,1], [relu, relu, identity], false, 0.1, 0.1, Flux.Optimise.Optimiser(Adam(0.01, (0.9, 0.999)), ExpDecay(0.1)), 1, 347, 1000, 1)
result_ST = NN_train(data_ST_2nd, config1_ST, trained_model = model_1st)
NN_results(config1_ST, result_ST)

model_2nd = result_ST.model
BSON.@save "ST_model_2nd.bson" model_2nd
BSON.@load "ST_model_2nd.bson" model_2nd

L_bounds = vcat(Float32.(sampling_config_2nd.lb), fill(Float32(-1e6), sum(config1_ST.layer[2:end])))
U_bounds = vcat(Float32.(sampling_config_2nd.ub), fill(Float32(1e6), sum(config1_ST.layer[2:end])))

MILP_model_2nd = rebuild_JuMP_Model(model_2nd, MILP_model_1st, config1_ST.freeze, L_bounds, U_bounds)
optimize!(MILP_model_2nd)
f_hat_2nd, x_star_2nd, gap_2nd = solution_evaluate(MILP_model_2nd, styblinski_tang)

#-----strategy 2: error based resampling

x_belows, x_aboves = find_max_errs(data_ST_1st_eb, model_1st, x_star_1st_eb)

# generate new samples around x_star
sampling_config_2nd_eb = generate_resample_config(sampling_config_1st_eb, x_star_1st_eb, 1.05, ("fixed_percentage_density", 1.0), "error_based", x_below = x_belows, x_above = x_aboves)
data_ST_2nd_eb = generate_data(styblinski_tang, sampling_config_2nd_eb, SobolSample(), 0.8)

config1_ST = NN_Config([10,512,256,1], [relu, relu, identity], false, 0.1, 0.1, Flux.Optimise.Optimiser(Adam(0.01, (0.9, 0.999)), ExpDecay(0.1)), 1, round(Int, sampling_config_2nd_eb.n_samples*0.8), 1000, 1)
result_ST = NN_train(data_ST_2nd_eb, config1_ST, trained_model = model_1st)
NN_results(config1_ST, result_ST)

model_2nd_eb = result_ST.model

L_bounds = vcat(Float32.(sampling_config_2nd_eb.lb), fill(Float32(-1e6), sum(config1_ST.layer[2:end])))
U_bounds = vcat(Float32.(sampling_config_2nd_eb.ub), fill(Float32(1e6), sum(config1_ST.layer[2:end])))
MILP_model_2nd_eb = rebuild_JuMP_Model(model_2nd_eb, MILP_model_1st_eb, config1_ST.freeze, L_bounds, U_bounds)
warmstart_JuMP_Model(MILP_model_2nd_eb, x_star_1st_eb)
optimize!(MILP_model_2nd_eb)

f_hat_2nd_eb, f_true_2nd_eb, x_star_2nd_eb, gap_2nd_eb = solution_evaluate(MILP_model_2nd_eb, styblinski_tang)

#-----strategy 3: sagmented error based resampling

x_belows, x_aboves = find_max_segmented_errs(data_ST_1st_seb, model_1st_seb, x_star_1st_seb, 20)
# generate new samples around x_star
sampling_config_2nd_seb = generate_resample_config(sampling_config_1st_seb, x_star_1st_seb, 1.05, ("fixed_percentage_density", 1.0), "segmented_error", x_below = x_belows, x_above = x_aboves)

data_ST_2nd_seb = generate_data(styblinski_tang, sampling_config_2nd_seb, SobolSample(), 0.8)
config1_ST = NN_Config([10,512,256,1], [relu, relu, identity], false, 0.1, 0.1, Flux.Optimise.Optimiser(Adam(0.01, (0.9, 0.999)), ExpDecay(0.1)), 1, round(Int, sampling_config_2nd_seb.n_samples*0.8), 1000, 1)
result_ST = NN_train(data_ST_2nd_seb, config1_ST, trained_model = model_1st_seb)
NN_results(config1_ST, result_ST)

model_2nd_seb = result_ST.model

L_bounds = vcat(Float32.(sampling_config_2nd_seb.lb), fill(Float32(-1e6), sum(config1_ST.layer[2:end])))
U_bounds = vcat(Float32.(sampling_config_2nd_seb.ub), fill(Float32(1e6), sum(config1_ST.layer[2:end])))
MILP_model_2nd_seb = rebuild_JuMP_Model(model_2nd_seb, MILP_model_1st_seb, config1_ST.freeze, L_bounds, U_bounds)
warmstart_JuMP_Model(MILP_model_2nd_seb, x_star_1st_seb)
optimize!(MILP_model_2nd_seb)
f_hat_2nd_seb, f_true_2nd_seb, x_star_2nd_seb, gap_2nd_seb = solution_evaluate(MILP_model_2nd_seb, styblinski_tang)

