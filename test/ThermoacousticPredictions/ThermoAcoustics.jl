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
ArgRs = 0; ArgRn = pi   # Phase of reflection coefficient: ArgRs (inlet), ArgRn (outlet)

# [training_Y(cal_index,1),training_Y(cal_index,2)] = Helmholtz_gain_phase('Secant',G(cal_index),phi(cal_index),R_in(cal_index),ArgRs,R_out(cal_index),ArgRn,alpha(cal_index),config_index,s_init)
compute_growth_rate(x::NTuple{5, Float64}) = mat"Helmholtz_gain_phase('Secant', $(x[1]), $(x[2]), $(x[3]), 0, $(x[4]), pi, $(x[5]), 11, 1i*112*2*pi)"

#--------------------------------- Initial training ---------------------------------#
L_bounds_init = [0.5, 0, 0.7, 0.6, 100]
U_bounds_init = [3, pi, 1, 1, 160]
sampling_config_init = Sampling_Config(1000, L_bounds_init, U_bounds_init)

# data_flame = load_data(joinpath(@__DIR__, "data_init.csv"), 0.8, 5, 1)

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
NN_results(config1_flame, result_flame)
plot_learning_curve(config1_flame, result_flame.err_hist)

model_init = result_flame.model
BSON.@save joinpath(@__DIR__, "surrogate_init.bson") model_init
BSON.@load joinpath(@__DIR__, "surrogate_init.bson") model_init

# convert the trained nerual net to a JuMP model
@info "option 1: weak big-M"
# bounds of the layers (arbitrary large big-M)
L_bounds_init_bigM = vcat(Float32.((sampling_config_init.lb .- vec(mean_init)) ./ vec(std_init)), fill(Float32(-500), sum(config1_flame.layer[2:end])))
U_bounds_init_bigM = vcat(Float32.((sampling_config_init.ub .- vec(mean_init)) ./ vec(std_init)), fill(Float32(500), sum(config1_flame.layer[2:end])))

MILP_model = JuMP_Model(model_init, L_bounds_init_bigM, U_bounds_init_bigM)
write_to_file(MILP_model, joinpath(@__DIR__, "MILP_init.mps"))
MILP_model_raw = read_from_file("MILP_init.mps")

set_optimizer_attribute(MILP_model, "PoolSolutions", 10)      # PoolSolutions = 10 (default)
set_optimizer_attribute(MILP_model, "LogFile", joinpath(@__DIR__, "gurobi_log_init_bigM.log"))
optimize!(MILP_model)

f_hat, f_true, x_star_init, x_star_init_norm, gap = solution_evaluate(MILP_model, compute_growth_rate, mean = mean_init, std = std_init)

num_solutions_init_bigM = MOI.get(MILP_model, MOI.ResultCount())
sol_pool_x_init_bigM, _ = sol_pool(MILP_model, num_solutions_init_bigM, mean = mean_init, std = std_init)

plot_dual_contours(data_flame_norm, model_init, x_star_init_norm, "sol_pool", sol_pool_x_init_bigM, [1,2])

@info "option 2: bound tightening and compression (Gogeta.jl)"
MILP_bt = Model()
set_optimizer(MILP_bt, Gurobi.Optimizer)
set_silent(MILP_bt)
set_attribute(MILP_bt, "TimeLimit", 10)
# current compressed model has no dropout layer but the original model has dropout layer
compressed_model, removed_neurons, bounds_U, bounds_L = NN_formulate!(MILP_bt, model_init, sampling_config_init_norm.ub, sampling_config_init_norm.lb; bound_tightening="standard", compress=true, silent=false);

@objective(MILP_bt, Min, MILP_bt[:x][4,1])
set_attribute(MILP_bt, "TimeLimit", 1800)
unset_silent(MILP_bt)
set_optimizer_attribute(MILP_bt, "LogFile", joinpath(@__DIR__, "gurobi_log_init_bt.log"))
optimize!(MILP_bt)

# MPS (Mathematical Programming System) File Format
write_to_file(MILP_bt, joinpath(@__DIR__, "MILP_init_bt.mps"))

f_hat_bt, f_true_bt, x_star_init_bt, x_star_init_norm_bt, gap_bt = solution_evaluate(MILP_bt, compute_growth_rate, mean = mean_init, std = std_init)
x_star_init_bt

# store multiple solutions in the solution pool
num_solutions_init_bt = MOI.get(MILP_bt, MOI.ResultCount())
sol_pool_x_init_bt, _ = sol_pool(MILP_bt, num_solutions_init_bt, mean = mean_init, std = std_init)

# Visualise the surrogate model using all data points and the solution pool
plot_dual_contours(data_flame_norm, model_init, x_star_init_norm, "sol_pool", sol_pool_x_init_bt, [1,2])

#------------ apply Monte Carlo Dropout

#----- 1. Uncertainty
# take the sample standard deviation as the estimate of the uncertainty
pred, pred_dist, means, stds, x_top_std = predict_dist(data_flame_norm, model_init, 100, 10)
pred_point, sd = predict_point(data_flame_norm, model_init, 100)
plot_dual_contours(data_flame_norm, model_init, x_star_init_norm_bt, "x_top_std", [col for col in eachcol(x_top_std)], [1,2])
plot_single_contour(data_flame_norm, model_init, x_star_init_norm_bt, "Uncertainty of the Predictions", vec(stds), "x_top_std", [col for col in eachcol(x_top_std)], [1,2])

# plot the predictive distribution of the 5th entry of x_test
println(data_flame_norm.x_test[:, 5])
kdeplot(pred_dist[5], means[5])

# remove the outliers
pred_rm = remove_outliers(data_flame_norm, pred_dist)
kdeplot(pred_rm[5], pred_point[5])

# take the sample standard deviation of flitered predictions as the estimate of the uncertainty
pred_dist_rm, means_rm, stds_rm, x_top_std_rm = predict_dist_filtered(data_flame_norm, model_init, 100, 10)
plot_dual_contours(data_flame_norm, model_init, x_star_init_norm_bt, "x_top_std_rm", [col for col in eachcol(x_top_var_rm)], [1,2])
plot_single_contour(data_flame_norm, model_init, x_star_init_norm_bt, "Uncertainty of the Predictions", vec(stds_rm), "x_top_std_rm", [col for col in eachcol(x_top_std_rm)], [1,2])

#------ 2. Prediction error
pred_ee, x_top_ee = expected_prediction_error(data_flame_norm, model_init, 100, 10)
kdeplot(pred_ee, mean(pred_ee))
histogram(data_flame_norm.x_test[1, :], weights = pred_ee, bins = 200, xlabel = "x_test[1]", ylabel = "pred_ee", legend=false)

plot_dual_contours(data_flame_norm, model_init, x_star_init_norm_bt, "x_top_ee", [col for col in eachcol(x_top_ee)], [1,2])
plot_single_contour(data_flame_norm, model_init, x_star_init_norm_bt, "Expected Prediction Error", vec(pred_ee), "x_top_var", [col for col in eachcol(x_top_var)], [1,2])

#------ 3. Expected improvement
pred_ei, x_top_ei = expected_improvement(data_flame_norm, model_init, x_star_init_norm_bt, 100, 50, true, 10.)
kdeplot(pred_ei, mean(pred_ei))

plot_dual_contours(data_flame_norm, model_init, x_star_init_norm_bt, "x_top_ei", [col for col in eachcol(x_top_ei)], [1,2])
plot_single_contour(data_flame_norm, model_init, x_star_init_norm_bt, "Expected Improvement", vec(pred_ei), "x_top_var", [col for col in eachcol(x_top_var)], [1,2])
plot_single_contour(data_flame_norm, model_init, x_star_init_norm_bt, "Uncertainty of the Predictions", vec(stds), "x_top_ei", [col for col in eachcol(x_top_ei)], [1,2])

#--------------------------------- 1st iteration ---------------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_1st, sampling_config_1st = generate_resample_configs_mc(sampling_config_init_norm, x_top_std_rm, 0.05, 0.3, mean_init, std_init)
data_flame_1st_new = generate_and_combine_data(compute_growth_rate, sampling_configs_1st, HaltonSample(), 0.8)
data_flame_1st = combine_datasets(data_flame_1st_new, data_flame)

Plots.scatter(data_flame_1st_new.x_train[1, :], data_flame_1st_new.x_train[2, :], color = :lightgreen, xlabel="x₁", ylabel="x₂", legend=:bottomright, label="New Generated Data")
Plots.scatter!(data_flame.x_train[1, :], data_flame.x_train[2, :], color = :viridis, legend=:bottomright, label="Old Training Data")
Plots.scatter!(data_flame.x_test[1, :], data_flame.x_test[2, :], color = :orange, legend=:bottomright, label="Old Test Data")

Plots.scatter(data_flame_1st.x_train[1, :], data_flame_1st.x_train[2, :], color = :viridis, xlabel="x₁", ylabel="x₂", legend=:bottomright, label="Training Data")
Plots.scatter!(data_flame_1st.x_test[1, :], data_flame_1st.x_test[2, :], color = :orange, legend=:bottomright, label="Test Data")

@info "option 1: train with the whole combined data"

data_flame_1st_norm, mean_1st, std_1st = normalise_data(data_flame_1st)
sampling_config_1st_norm = Sampling_Config(
    sampling_config_1st.n_samples,
    (sampling_config_1st.lb .- vec(mean_1st)) ./ vec(std_1st),
    (sampling_config_1st.ub .- vec(mean_1st)) ./ vec(std_1st)
)

config1_flame = NN_Config([5,256,256,128,1], [relu, relu, relu, identity], false, 0, 0.2, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(0.1)), 1, 800, 1000, 0)

config1_flame_1st = NN_Config([5,256,256,128,1], [relu, relu, relu, identity], false, 0, 0.2, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(0.1)), 1, 
    size(data_flame_norm_1st_norm.x_train,2), 1000, 1)
train_time = @elapsed result_flame_1st = NN_train(data_flame_norm_1st_norm, config1_flame_1st, trained_model = model_init)
NN_results(config1_flame_1st, result_flame_1st)
plot_learning_curve(config1_flame_1st, result_flame_1st.err_hist)

model_1st = result_flame_1st.model
BSON.@save joinpath(@__DIR__, "surrogate_1st.bson") model_1st
BSON.@load joinpath(@__DIR__, "surrogate_1st.bson") model_1st

# convert the trained nerual net to a JuMP model using bound tightening and compression (Gogeta.jl)
MILP_bt_1st = Model()
set_optimizer(MILP_bt_1st, Gurobi.Optimizer)
set_silent(MILP_bt_1st)
set_attribute(MILP_bt_1st, "TimeLimit", 10)

# current compressed model has no dropout layer but the original model has dropout layer
compressed_model_1st, removed_neurons_1st, bounds_U_1st, bounds_L_1st = NN_formulate!(MILP_bt_1st, model_1st, sampling_config_1st_norm.ub, sampling_config_1st_norm.lb; bound_tightening="standard", compress=true, silent=false);

@objective(MILP_bt_1st, Min, MILP_bt_1st[:x][4,1])
set_attribute(MILP_bt_1st, "TimeLimit", 3600)
unset_silent(MILP_bt_1st)
set_attribute(MILP_bt_1st, "LogFile", joinpath(@__DIR__, "gurobi_log_1st.log"))
# warmstart_JuMP_Model(MILP_bt_1st, x_star_init_norm_bt)
optimize!(MILP_bt_1st)

write_to_file(MILP_bt_1st, joinpath(@__DIR__, "flame_1st_bt.mps"))

f_hat_1st, f_true_1st, x_star_1st, x_star_1st_norm, gap_1st = solution_evaluate(MILP_bt_1st, compute_growth_rate, mean = mean_1st, std = std_1st)

@info "option 2: train with the filtered combined data"
data_flame_1st_filtered = filter_data_within_bounds(data_flame_1st, sampling_config_1st.lb, sampling_config_1st.ub)
data_flame_1st_filtered_norm, mean_1st_filtered, std_1st_filtered = normalise_data(data_flame_1st_filtered)
sampling_config_1st_filtered_norm = Sampling_Config(
    sampling_config_1st.n_samples,
    (sampling_config_1st.lb .- vec(mean_1st_filtered)) ./ vec(std_1st_filtered),
    (sampling_config_1st.ub .- vec(mean_1st_filtered)) ./ vec(std_1st_filtered)
)

config1_flame_1st_filtered = NN_Config([5,256,256,128,1], [relu, relu, relu, identity], false, 0, 0.2, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(0.1)), 1, 
    size(data_flame_1st_filtered_norm.x_train,2), 1000, 0)
train_time = @elapsed result_flame_1st_filtered = NN_train(data_flame_1st_filtered_norm, config1_flame_1st_filtered, trained_model = model_init)
NN_results(config1_flame_1st_filtered, result_flame_1st_filtered)

model_1st_filtered = result_flame_1st_filtered.model
BSON.@save joinpath(@__DIR__, "surrogate_1st_filtered.bson") model_1st_filtered
BSON.@load joinpath(@__DIR__, "surrogate_1st_filtered.bson") model_1st_filtered

# convert the trained nerual net to a JuMP model using bound tightening and compression (Gogeta.jl)
MILP_bt_1st_filtered = Model()
set_optimizer(MILP_bt_1st_filtered, Gurobi.Optimizer)
set_silent(MILP_bt_1st_filtered)
set_attribute(MILP_bt_1st_filtered, "TimeLimit", 10)

# current compressed model has no dropout layer but the original model has dropout layer
compressed_model_1st_filtered, removed_neurons_1st_filtered, bounds_U_1st_filtered, bounds_L_1st_filtered = NN_formulate!(MILP_bt_1st_filtered, model_1st_filtered, 
    sampling_config_1st_filtered_norm.ub, sampling_config_1st_filtered_norm.lb; bound_tightening="standard", compress=true, silent=false)

@objective(MILP_bt_1st_filtered, Min, MILP_bt_1st_filtered[:x][4,1])
set_attribute(MILP_bt_1st_filtered, "TimeLimit", 3600)
set_attribute(MILP_bt_1st_filtered, "OutputFlag", 1)    # unset_silent(MILP_bt_1st_filtered)
set_attribute(MILP_bt_1st_filtered, "LogFile", joinpath(@__DIR__, "gurobi_log_1st_filtered.log"))
# log_filename = "gurobi_log_1st_filtered_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
# set_attribute(MILP_bt_1st_filtered, "LogFile", joinpath(@__DIR__, log_filename))
# warmstart_JuMP_Model(MILP_bt_1st_filtered, x_star_init_norm_bt)

optimize!(MILP_bt_1st_filtered)

write_to_file(MILP_bt_1st_filtered, joinpath(@__DIR__, "flame_1st_bt_filtered.mps"))

f_hat_1st_filtered, f_true_1st_filtered, x_star_1st_filtered, x_star_1st_norm_filtered, gap_1st_filtered = solution_evaluate(MILP_bt_1st_filtered, compute_growth_rate, mean = mean_1st_filtered, std = std_1st_filtered)
