using MATLAB
using Statistics
using Surrogates
using Flux
using CSV, DataFrames
using BSON
using Plots
using JuMP, Gurobi
using Random

include("../../src/NNSurrogate.jl")
include("../../src/NNJuMP.jl")
include("../../src/NNGogeta.jl")
include("../../src/NNOptimise.jl")
include("../../src/MCDropout.jl")

"""
- Flame model with acoustic tools (5 variables)
"""

Random.seed!(1)

mat"addpath(genpath(fullfile(pwd, 'test', 'ThermoacousticPredictions')))"  # add the path of the MATLAB code
ArgRs=0; ArgRn=pi   # Phase of reflection coefficient: ArgRs (inlet), ArgRn (outlet)

# [training_Y(cal_index,1),training_Y(cal_index,2)] = Helmholtz_gain_phase('Secant',G(cal_index),phi(cal_index),R_in(cal_index),ArgRs,R_out(cal_index),ArgRn,alpha(cal_index),config_index,s_init)
compute_growth_rate(x::NTuple{5, Float64}) = mat"Helmholtz_gain_phase('Secant', $(x[1]), $(x[2]), $(x[3]), 0, $(x[4]), pi, $(x[5]), 11, 1i*112*2*pi)"

#--------------------------------- Initial training ---------------------------------#
L_bounds_init = [0.5, 0, 0.7, 0.6, 100]
U_bounds_init = [3, pi, 1, 1, 160]
sampling_config_init = Sampling_Config(1000, L_bounds_init, U_bounds_init)

# get the location of the script file
root = dirname(@__FILE__)

# a robust representation of the filepath to data file
csv_file_path = joinpath(root, "data_init.csv")

# read the data from the csv file
rdata = DataFrame(CSV.File(csv_file_path, header=false))

# Extract x (first 5 columns) and y (6th column)
x = Matrix(rdata[:, 1:5])'
y = reshape(rdata[:, 6], :, 1)'

# split the data into train set and test set
train_data, test_data = Flux.splitobs((x, y), at = 0.8)

# convert the data to the format of NN_Data
data_flame = NN_Data()
data_flame.x_train = Float32.(train_data[1])
data_flame.y_train = Float32.(train_data[2])
data_flame.x_test = Float32.(test_data[1])
data_flame.y_test = Float32.(test_data[2])

# normailse the data
data_flame_norm, mean_init, std_init = normalise_data(data_flame)
sampling_config_init_norm = Sampling_Config(
    sampling_config_init.n_samples,
    (sampling_config_init.lb .- vec(mean_init)) ./ vec(std_init),
    (sampling_config_init.ub .- vec(mean_init)) ./ vec(std_init))
Plots.scatter(data_flame_norm.x_train[1, :], data_flame_norm.x_train[2, :], color = :viridis, xlabel="x₁", ylabel="x₂", legend=:bottomright, label="Training Data")
Plots.scatter!(data_flame_norm.x_test[1, :], data_flame_norm.x_test[2, :], color = :orange, legend=:bottomright, label="Test Data")

config1_flame = NN_Config([5,256,256,128,1], [relu, relu, relu, identity], false, 0, 0.2, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(0.1)), 1, 800, 1000, 0)

# trian the nerual net
train_time = @elapsed result_flame = NN_train(data_flame_norm, config1_flame)
plot_learning_curve(config1_flame, result_flame.err_hist)

model_init = result_flame.model
BSON.@save joinpath(@__DIR__, "surrogate_init.bson") model_init
BSON.@load joinpath(@__DIR__, "surrogate_init.bson") model_init

# convert the trained nerual net to a JuMP model
@info "option 1: weak big-M"
# bounds of the layers (arbitrary large big-M)
L_bounds = vcat(Float32.((sampling_config_init.lb .- vec(mean_init)) ./ vec(std_init)), fill(Float32(-500), sum(config1_flame.layer[2:end])))
U_bounds = vcat(Float32.((sampling_config_init.ub .- vec(mean_init)) ./ vec(std_init)), fill(Float32(500), sum(config1_flame.layer[2:end])))

MILP_model = JuMP_Model(model_init, L_bounds, U_bounds)
write_to_file(MILP_model, "MILP_init.mps")
MILP_model_raw = read_from_file("MILP_init.mps")

set_optimizer_attribute(model, "PoolSolutions", 5)      # PoolSolutions = 10 (default)
optimize!(MILP_model)

f_hat, f_true, x_star_init, x_star_init_norm, gap = solution_evaluate(MILP_model, compute_growth_rate, mean = mean_init, std = std_init)

@info "option 2: bound tightening and compression (Gogeta.jl)"
MILP_bt = Model()
set_optimizer(MILP_bt, Gurobi.Optimizer)
set_silent(MILP_bt)
set_attribute(MILP_bt, "TimeLimit", 10)
# current compressed model has no dropout layer but the original model has dropout layer
compressed_model, removed_neurons, bounds_U, bounds_L = NN_formulate!(MILP_bt, model_init, U_bounds_init, L_bounds_init; bound_tightening="standard", compress=true, silent=false);

@objective(MILP_bt, Min, MILP_bt[:x][4,1])
set_attribute(MILP_bt, "TimeLimit", 1000)
unset_silent(MILP_bt)
set_optimizer_attribute(MILP_bt, "LogFile", joinpath(@__DIR__, "gurobi_log_init.log"))
optimize!(MILP_bt)

write_to_file(MILP_bt, joinpath(@__DIR__, "MILP_init_bt.mps"))

f_hat_bt, f_true_bt, x_star_init_bt, x_star_init_norm_bt, gap_bt = solution_evaluate(MILP_bt, compute_growth_rate, mean = mean_init, std = std_init)

# store multiple solutions in the solution pool
num_solutions = MOI.get(MILP_bt, MOI.ResultCount())
sol_pool_x, _ = sol_pool(MILP_bt, num_solutions, mean = mean_init, std = std_init)

# Visualise the surrogate model using all data points and the solution pool
plot_tricontour_with_solpool(data_flame_norm, model_init, x_star_init_norm_bt, sol_pool_x, [2,3])
sol_pool_x

#------------ apply Monte Carlo Dropout

#----- 1. Uncertainty
pred_dist, means, stds, x_top_var = predict_dist(data_flame_norm, model_init, 100, 50)
pred_point, sd = predict_point(data_flame_norm, model_init, 100)
plot_contour(data_flame_norm, model_init, x_star_init_norm_bt, "x_top_var", [col for col in eachcol(x_top_var)], [1,2])
plot_contour_pure(data_flame_norm, model_init, x_star_init_norm_bt, "Uncertainty of the Predictions", vec(stds), [1,2])
y_true = collect(vec(hcat(data_flame.y_train, data_flame.y_test)))

# plot the predictive distribution of the 1st entry of x_test
kdeplot(pred_dist[2], means[2])

# pred_rm = remove_outliers(data_flame_norm, pred_dist)
# kdeplot(pred_rm[1], pred_point[1])
println(data_flame_norm.x_test[:, 2])

#------ 2. Prediction error
pred_ee, x_top_ee = expected_prediction_error(data_flame_norm, model_init, 100, 10)
kdeplot(pred_ee, mean(pred_ee))
histogram(data_flame_norm.x_test[1, :], weights = pred_ee, bins = 200, xlabel = "x_test[1]", ylabel = "pred_ee", legend=false)

plot_contour(data_flame_norm, model_init, x_star_init_norm_bt, "x_top_ee", [col for col in eachcol(x_top_ee)], [1,2])
plot_contour_pure(data_flame_norm, model_init, x_star_init_norm_bt, "Expected Prediction Error", vec(pred_ee), "x_top_var", [col for col in eachcol(x_top_var)], [1,2])

#------ 3. Expected improvement
pred_ei, x_top_ei = expected_improvement(data_flame_norm, model_init, x_star_init_norm_bt, 100, 50, true, 10.)
kdeplot(pred_ei, mean(pred_ei))

plot_contour(data_flame_norm, model_init, x_star_init_norm_bt, "x_top_ei", [col for col in eachcol(x_top_ei)], [1,2])
plot_contour_pure(data_flame_norm, model_init, x_star_init_norm_bt, "Expected Improvement", vec(pred_ei), "x_top_var", [col for col in eachcol(x_top_var)], [1,2])
plot_contour_pure(data_flame_norm, model_init, x_star_init_norm_bt, "Uncertainty of the Predictions", vec(stds), "x_top_ei", [col for col in eachcol(x_top_ei)], [1,2])

#--------------------------------- 1st iteration ---------------------------------#
# Resample densely around the points with the highest uncertainty
sampling_config_1st_norm, sampling_config_1st = generate_resample_configs_mc(sampling_config_init_norm, x_top_var, 0.05, 0.3, mean_init, std_init)
data_flame_norm_1st_new = generate_and_combine_data(compute_growth_rate, sampling_config_1st, HaltonSample(), 0.8)
data_flame_norm_1st = combine_datasets(data_flame_norm_1st_new, data_flame)

Plots.scatter(data_flame_norm_1st_new.x_train[1, :], data_flame_norm_1st_new.x_train[2, :], color = :lightgreen, xlabel="x₁", ylabel="x₂", legend=:bottomright, label="New Generated Data")
Plots.scatter!(data_flame.x_train[1, :], data_flame.x_train[2, :], color = :viridis, legend=:bottomright, label="Old Training Data")
Plots.scatter!(data_flame.x_test[1, :], data_flame.x_test[2, :], color = :orange, legend=:bottomright, label="Old Test Data")

Plots.scatter(data_flame_norm_1st.x_train[1, :], data_flame_norm_1st.x_train[2, :], color = :viridis, xlabel="x₁", ylabel="x₂", legend=:bottomright, label="Training Data")
Plots.scatter!(data_flame_norm_1st.x_test[1, :], data_flame_norm_1st.x_test[2, :], color = :orange, legend=:bottomright, label="Test Data")

data_flame_norm_1st_norm, mean_1st, std_1st = normalise_data(data_flame_norm_1st)

config1_flame = NN_Config([5,256,256,128,1], [relu, relu, relu, identity], false, 0, 0.2, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(0.1)), 1, 800, 1000, 0)

config1_flame_1st = NN_Config([5,256,256,128,1], [relu, relu, relu, identity], false, 0, 0.2, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(0.1)), 1, 
    size(data_flame_norm_1st_norm.x_train,2), 5000, 1)
train_time = @elapsed result_flame_1st = NN_train(data_flame_norm_1st_norm, config1_flame_1st, trained_model = model_init)
NN_results(config1_flame_1st, result_flame_1st)
plot_learning_curve(config1_flame_1st, result_flame_1st.err_hist)

model_1st = result_flame_1st.model
BSON.@save "surrogate_1st.bson" model_1st
BSON.@load "surrogate_1st.bson" model_1st

# convert the trained nerual net to a JuMP model
@info "option 2: bound tightening and compression (Gogeta.jl)"
MILP_bt_1st = Model()
set_optimizer(MILP_bt_1st, Gurobi.Optimizer)
set_silent(MILP_bt_1st)
set_attribute(MILP_bt_1st, "TimeLimit", 10)

L_bounds_1st = (L_bounds_init .- vec(mean_1st)) ./ vec(std_1st)
U_bounds_1st = (U_bounds_init .- vec(mean_1st)) ./ vec(std_1st)
# current compressed model has no dropout layer but the original model has dropout layer
compressed_model_1st, removed_neurons_1st, bounds_U_1st, bounds_L_1st = NN_formulate!(MILP_bt_1st, model_1st, U_bounds_1st, L_bounds_1st; bound_tightening="standard", compress=true, silent=false);

@objective(MILP_bt_1st, Min, MILP_bt_1st[:x][4,1])
set_attribute(MILP_bt_1st, "TimeLimit", 3600)
unset_silent(MILP_bt_1st)
set_attribute(MILP_bt_1st, "LogFile", joinpath(@__DIR__, "gurobi_log_1st.log"))
warmstart_JuMP_Model(MILP_bt_1st, x_star_init_norm_bt)
optimize!(MILP_bt_1st)

write_to_file(MILP_bt_1st, "flame_1st_bt.mps")

f_hat_1st, f_true_1st, x_star_1st, x_star_1st_norm, gap_1st = solution_evaluate(MILP_bt_1st, compute_growth_rate, mean = mean_1st, std = std_1st)
