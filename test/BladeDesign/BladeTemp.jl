using Statistics
using Surrogates
using Flux
using JuMP
using Gurobi
using MATLAB
using DataFrames, CSV
using BSON
using Dates
using Random

include("../../src/NNSurrogate.jl")
include("../../src/NNJuMP.jl")
include("../../src/NNGogeta.jl")
include("../../src/NNOptimise.jl")
include("../../src/MCDropout.jl")

"""
- Blade max-temperature simulation model (6 variables)
"""

mat"addpath(pwd(), '../ThermalAnalysisOfJetEngineTurbineBlade')"  # add the path of the MATLAB code

# arguments: T_air, T_gas, h_air, h_gas_pressureside, h_gas_suctionside, h_gas_tip
blade_max_temp(x::NTuple{6, Float64}) = mat"computeMaxTemp($(x[1]), $(x[2]), $(x[3]), $(x[4]), $(x[5]), $(x[6]))"

#--------------------------------- Initial training ---------------------------------#

L_bounds_init = [120, 900, 20, 40, 30, 10]
U_bounds_init = [180, 1200, 40, 60, 50, 30]
sampling_config_init = Sampling_Config(1000, L_bounds_init, U_bounds_init)

# data_temp = generate_data(blade_max_temp, sampling_config_init, SobolSample(), 0.8)
root = dirname(@__FILE__)
csv_file_path = joinpath(root, "combined_data.csv")

data_temp = load_data(csv_file_path, 0.8, 6, 1)

# normalise the data, including the outputs
# data_temp_norm, mean_init, std_init = normalise_data(data_temp, false)

# sampling_config_init_norm = Sampling_Config(
#     sampling_config_init.n_samples,
#     (sampling_config_init.lb .- vec(mean_init)) ./ vec(std_init),
#     (sampling_config_init.ub .- vec(mean_init)) ./ vec(std_init)
# )

# train the surrogate model
config_temp = NN_Config([6,50,50,1], [relu, relu, identity], false, 0.0, 0.0, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, 800, 1000, 0)

train_time = @elapsed result_temp = NN_train(data_temp, config_temp)
NN_results(config_temp, result_temp)
plot_learning_curve(config_temp, result_temp.err_hist)

model_init = result_temp.model
BSON.@save joinpath(@__DIR__, "surrogate_init.bson") model_init
BSON.@load joinpath(@__DIR__, "surrogate_init.bson") model_init

#-----------Gogeta------------#
@info "bound tightening and compression (Gogeta.jl)"
MILP_bt = Model()
set_optimizer(MILP_bt, Gurobi.Optimizer)
set_silent(MILP_bt)
set_attribute(MILP_bt, "TimeLimit", 10)
build_time = @elapsed compressed_model, removed_neurons, bounds_U, bounds_L = NN_formulate!(MILP_bt, model_init, sampling_config_init.ub, sampling_config_init.lb; bound_tightening="fast", compress=true, silent=false);

@objective(MILP_bt, Min, MILP_bt[:x][3,1])

set_attribute(MILP_bt, "TimeLimit", 1800)
unset_silent(MILP_bt)
log_filename = "gurobi_log_init_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt)
write_to_file(MILP_bt, joinpath(@__DIR__, "model_init_bt.mps"))
MILP_bt = read_from_file(joinpath(@__DIR__, "model_init_bt.mps"))

x_star_init = [value.(MILP_bt[:x][0,i]) for i in 1:length(MILP_bt[:x][0,:])]
# x_star_init_norm = [value.(MILP_bt[:x][0,i]) for i in 1:length(MILP_bt[:x][0,:])] .* std_init .+ mean_init

solution_evaluate(MILP_bt, blade_max_temp)

# store multiple solutions in the solution pool
num_solutions_init_bt = MOI.get(MILP_bt, MOI.ResultCount())
sol_pool_x_init_bt, _ = sol_pool(MILP_bt, num_solutions_init_bt)

# visualise the surrogate model 
fig = plot_dual_contours(data_temp, model_init, x_star_init, "sol_pool", sol_pool_x_init_bt, [1,2], 1)
Makie.save(joinpath(root, "images/exp1_init_dual_sol_pool.png"), fig)

#------------ apply Monte Carlo Dropout to the surrogate model ------------#
config_temp_dp = NN_Config([6,50,50,1], [relu, relu, identity], false, 0.0, 0.1, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, 800, 1000, 0)
train_time = @elapsed result_temp_dp = NN_train(data_temp, config_temp_dp)
NN_results(config_temp_dp, result_temp_dp)
model_init_dp = result_temp.model

pred, pred_dist, means, stds, x_top_std = predict_dist(data_temp, model_init_dp, 100, 10)
fig = plot_dual_contours(data_temp, model_init, x_star_init, "points with top 10 highest uncertainty", [col for col in eachcol(x_top_std)], [1,2], 1)
Makie.save(joinpath(root, "images/exp1_init_dual_top_std.pdf"), fig)

# plot the predictive distribution of the 5th entry of x_test
pred_point, _ = predict_point(data_temp, model_init_dp, 100, 1)
println(data_temp.x_test[:, 5])
kdeplot(pred_dist[5][:,1], means[1,5])
savefig(joinpath(root, "images/exp1_Predict_Distribution_5th_Entry_x_test.svg"))
# remove the outliers
# pred_rm = remove_outliers_per_dist(pred_dist[5][:,10])
# kdeplot(pred_rm, pred_point[5])

fig = plot_single_contour(data_temp, model_init_dp, x_star_init, "Uncertainty of the Predictions", vec(stds),"x_top_std", [col for col in eachcol(x_top_std)], [1,2])
Makie.save(joinpath(root, "images/exp1_Uncertainty_10.svg"), fig)

#------------------------------ 1st iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_1st, sampling_config_1st = generate_resample_configs_mc(sampling_config_init, [x_top_std x_star_init hcat(sol_pool_x_init_bt...)[:, 2:end]], 0.10, 0.3, zeros(Float64, 6, 1), ones(Float64, 6, 1))

data_temp_1st_new = generate_and_combine_data(blade_max_temp, sampling_configs_1st, SobolSample(), 0.8)
data_temp_1st = combine_datasets(data_temp, data_temp_1st_new)
data_temp_1st_filtered = filter_data_within_bounds(data_temp_1st, sampling_config_1st.lb, sampling_config_1st.ub)

x_1st_added = hcat(data_temp_1st_new.x_train, data_temp_1st_new.x_test)

Plots.scatter(x_1st_added[1, :], x_1st_added[2, :], color = :lightgreen, xlabel="x₁", ylabel="x₂", legend=:bottomright, label="Resampled Data")
Plots.scatter!(data_temp.x_train[1, :], data_temp.x_train[2, :], color = :viridis, xlabel="x₁", ylabel="x₂", legend=:bottomright, label="Initial Training Data")
Plots.scatter!(data_temp.x_test[1, :], data_temp.x_test[2, :], color = :orange, legend=:bottomright, label="Initial Test Data")
vline!([sampling_config_1st.lb[1],sampling_config_1st.ub[1]], label="x₁ bounds", linestyle=:dashdot, color=:red, linewidth = 2)
hline!([sampling_config_1st.lb[2],sampling_config_1st.ub[2]], label="x₂ bounds", linestyle=:dashdot, color=:purple, linewidth = 2)
savefig(joinpath(root, "images/exp1_1st_scattered_point.svg"))

println("# New added samples: ", sampling_config_1st.n_samples)
println(" # Previous samples: " ,sampling_config_init.n_samples)
println(" # Filtered samples: ", size(data_temp_1st_filtered.x_train)[2] + size(data_temp_1st_filtered.x_test)[2])

# data_temp_1st_filtered_norm, mean_1st_filtered, std_1st_filtered, mean_1st_filtered_y, std_1st_filtered_y = normalise_data(data_temp_1st_filtered, true)

sampling_config_1st_filtered = Sampling_Config(
    sampling_config_1st.n_samples,
    sampling_config_1st.lb,
    sampling_config_1st.ub
)

config_temp_1st = NN_Config([6,50,50,1], [relu, relu, identity], false, 0.0, 0.0, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, size(data_temp_1st_filtered.x_train)[2], 1000, 1)
train_time_1st = @elapsed result_temp_1st = NN_train(data_temp_1st_filtered, config_temp_1st, trained_model = model_init)
NN_results(config_temp_1st, result_temp_1st)

model_1st = result_temp_1st.model
BSON.@save joinpath(@__DIR__, "surrogate_1st.bson") model_1st
BSON.@load joinpath(@__DIR__, "surrogate_1st.bson") model_1st

# convert the surrogate model to a MILP model
MILP_bt_1st = Model()
set_optimizer(MILP_bt_1st, Gurobi.Optimizer)
set_silent(MILP_bt_1st)
set_attribute(MILP_bt_1st, "TimeLimit", 10)

build_time = @elapsed compressed_model_1st, removed_neurons_1st, bounds_U_1st, bounds_L_1st = NN_formulate!(MILP_bt_1st, model_1st, sampling_config_1st_filtered.ub, sampling_config_1st_filtered.lb; bound_tightening="fast", compress=true, silent=false)
@objective(MILP_bt_1st, Min, MILP_bt_1st[:x][2,1])

set_attribute(MILP_bt_1st, "TimeLimit", 1800)
unset_silent(MILP_bt_1st)
log_filename = "gurobi_log_1st_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_1st, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_1st)
write_to_file(MILP_bt_1st, joinpath(@__DIR__, "model_1st_bt.mps"))

x_star_1st = [value.(MILP_bt_1st[:x][0,i]) for i in 1:length(MILP_bt_1st[:x][0,:])]
# x_star_1st = [value.(MILP_bt_1st[:x][0,i]) for i in 1:length(MILP_bt_1st[:x][0,:])] .* std_1st_filtered .+ mean_1st_filtered

solution_evaluate(MILP_bt_1st, blade_max_temp)

# store multiple solutions in the solution pool
num_solutions_1st_filtered = MOI.get(MILP_bt_1st, MOI.ResultCount())
sol_pool_x_1st_filtered, _ = sol_pool(MILP_bt_1st, num_solutions_1st_filtered)

# visualise the surrogate model
fig = plot_dual_contours(data_temp_1st_filtered, model_1st, x_star_1st, "sol_pool", sol_pool_x_1st_filtered, [1,2], 1)
Makie.save(joinpath(root, "images/exp1_1st_dual_sol_pool.png.png"), fig)

#------------ apply Monte Carlo Dropout to the surrogate model ------------#
config_temp_1st_dp = NN_Config([6,50,50,1], [relu, relu, identity], false, 0.0, 0.1, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, size(data_temp_1st_filtered.x_train)[2], 1000, 1)

train_time_1st_dp = @elapsed result_temp_1st_dp = NN_train(data_temp_1st_filtered, config_temp_1st_dp, trained_model = model_1st)
NN_results(config_temp_1st_dp, result_temp_1st_dp)
model_1st_dp = result_temp_1st_dp.model

pred_1st, pred_dist_1st, means_1st, stds_1st, x_top_std_1st = predict_dist(data_temp_1st_filtered, model_1st_dp, 100, 10)
fig = plot_dual_contours(data_temp_1st_filtered, model_1st, x_star_1st, "x_top_std", [col for col in eachcol(x_top_std_1st)], [1,2], 1)
Makie.save(joinpath(root, "images/exp1_1st_dual_top_std.png"), fig)

#------------------------------ 2nd iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_2nd, sampling_config_2nd = generate_resample_configs_mc(sampling_config_1st_filtered, [x_top_std_1st x_star_1st hcat(sol_pool_x_1st_filtered...)[:, 2:end]], 0.10, 0.3, zeros(Float64, 6, 1), ones(Float64, 6, 1))

data_temp_2nd_new = generate_and_combine_data(blade_max_temp, sampling_configs_2nd, SobolSample(), 0.8)
data_temp_2nd = combine_datasets(data_temp_1st_filtered, data_temp_2nd_new)
data_temp_2nd_filtered = filter_data_within_bounds(data_temp_2nd, sampling_config_2nd.lb, sampling_config_2nd.ub)

x_2nd_added = hcat(data_temp_2nd_new.x_train, data_temp_2nd_new.x_test)

Plots.scatter(x_2nd_added[1, :], x_2nd_added[2, :], color = :lightgreen, xlabel="x₁", ylabel="x₂", legend=:bottomright, label="Resampled Data")
Plots.scatter!(data_temp_1st_filtered.x_train[1, :], data_temp_1st_filtered.x_train[2, :], color = :viridis, xlabel="x₁", ylabel="x₂", legend=:bottomright, label="Training Data (1st iteration)")
Plots.scatter!(data_temp_1st_filtered.x_test[1, :], data_temp_1st_filtered.x_test[2, :], color = :orange, legend=:bottomright, label="Test Data (1st iteration)")
vline!([sampling_config_2nd.lb[1],sampling_config_2nd.ub[1]], label="x₁ bounds", linestyle=:dashdot, color=:red, linewidth = 2)
hline!([sampling_config_2nd.lb[2],sampling_config_2nd.ub[2]], label="x₂ bounds", linestyle=:dashdot, color=:purple, linewidth = 2)
savefig(joinpath(root, "images/exp1_2nd_scattered_point.svg"))

println("# New added samples: ", sampling_config_2nd.n_samples)
println(" # Previous samples: " ,sampling_config_1st_filtered.n_samples)
println(" # Filtered samples: ", size(data_temp_2nd_filtered.x_train)[2] + size(data_temp_2nd_filtered.x_test)[2])

sampling_config_2nd_filtered = Sampling_Config(
    sampling_config_2nd.n_samples,
    sampling_config_2nd.lb,
    sampling_config_2nd.ub
)

config_temp_2nd = NN_Config([6,50,50,1], [relu, relu, identity], false, 0.1, 0.0, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, size(data_temp_2nd_filtered.x_train)[2], 1000, 1)

train_time_2nd = @elapsed result_temp_2nd = NN_train(data_temp_2nd_filtered, config_temp_2nd, trained_model = model_1st)
NN_results(config_temp_2nd, result_temp_2nd)

model_2nd = result_temp_2nd.model
BSON.@save joinpath(@__DIR__, "surrogate_2nd.bson") model_2nd
BSON.@load joinpath(@__DIR__, "surrogate_2nd.bson") model_2nd

# convert the surrogate model to a MILP model
MILP_bt_2nd = Model()
set_optimizer(MILP_bt_2nd, Gurobi.Optimizer)
set_silent(MILP_bt_2nd)
set_attribute(MILP_bt_2nd, "TimeLimit", 10)

build_time = @elapsed compressed_model_2nd, removed_neurons_2nd, bounds_U_2nd, bounds_L_2nd = NN_formulate!(MILP_bt_2nd, model_2nd, sampling_config_2nd_filtered.ub, sampling_config_2nd_filtered.lb; bound_tightening="fast", compress=true, silent=false)
@objective(MILP_bt_2nd, Min, MILP_bt_2nd[:x][3,1])

set_attribute(MILP_bt_2nd, "TimeLimit", 1800)
unset_silent(MILP_bt_2nd)
log_filename = "gurobi_log_2nd_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_2nd, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_2nd)
write_to_file(MILP_bt_2nd, joinpath(@__DIR__, "model_2nd_bt.mps"))

x_star_2nd = [value.(MILP_bt_2nd[:x][0,i]) for i in 1:length(MILP_bt_2nd[:x][0,:])]

solution_evaluate(MILP_bt_2nd, blade_max_temp)

# store multiple solutions in the solution pool
num_solutions_2nd_filtered = MOI.get(MILP_bt_2nd, MOI.ResultCount())
sol_pool_x_2nd_filtered, _ = sol_pool(MILP_bt_2nd, num_solutions_2nd_filtered)

# visualise the surrogate model
fig = plot_dual_contours(data_temp_2nd_filtered, model_2nd, x_star_2nd, "sol_pool", sol_pool_x_2nd_filtered, [1,2], 1)
Makie.save(joinpath(root, "images/exp1_2nd_dual_sol_pool.png.png"), fig)

#------------ apply Monte Carlo Dropout to the surrogate model ------------#
config_temp_2nd_dp = NN_Config([6,50,50,1], [relu, relu, identity], false, 0.0, 0.1, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, size(data_temp_2nd_filtered.x_train)[2], 1000, 1)

train_time_2nd_dp = @elapsed result_temp_2nd_dp = NN_train(data_temp_2nd_filtered, config_temp_2nd_dp, trained_model = model_2nd)
NN_results(config_temp_2nd_dp, result_temp_2nd_dp)
model_2nd_dp = result_temp_2nd_dp.model

pred_2nd, pred_dist_2nd, means_2nd, stds_2nd, x_top_std_2nd = predict_dist(data_temp_2nd_filtered, model_2nd_dp, 100, 10)
fig = plot_dual_contours(data_temp_2nd_filtered, model_2nd, x_star_2nd, "x_top_std", [col for col in eachcol(x_top_std_2nd)], [1,2], 1)
Makie.save(joinpath(root, "images/exp1_2nd_dual_top_std.png"), fig)

#------------------------------ 3rd iteration --------------------------#

sampling_configs_3rd, sampling_config_3rd = generate_resample_configs_mc(sampling_config_2nd_filtered, [x_top_std_2nd x_star_2nd hcat(sol_pool_x_2nd_filtered...)[:, 2:end]], 0.10, 0.3, zeros(Float64, 6, 1), ones(Float64, 6, 1))

data_temp_3rd_new = generate_and_combine_data(blade_max_temp, sampling_configs_3rd, SobolSample(), 0.8)
data_temp_3rd = combine_datasets(data_temp_2nd_filtered, data_temp_3rd_new)
data_temp_3rd_filtered = filter_data_within_bounds(data_temp_3rd, sampling_config_3rd.lb, sampling_config_3rd.ub)

x_3rd_added = hcat(data_temp_3rd_new.x_train, data_temp_3rd_new.x_test)

Plots.scatter(x_3rd_added[1, :], x_3rd_added[2, :], color = :lightgreen, xlabel="x₁", ylabel="x₂", legend=:bottomright, label="Resampled Data", size = (530, 500), legendfontsize = 10, tickfontsize = 10, labelfontsize = 14)
Plots.scatter!(data_temp_2nd_filtered.x_train[1, :], data_temp_2nd_filtered.x_train[2, :], color = :viridis, xlabel="x₁", ylabel="x₂", legend=:bottomright, label="Training Data (2nd iteration)", legendfontsize = 10, tickfontsize = 10, labelfontsize = 14)
Plots.scatter!(data_temp_2nd_filtered.x_test[1, :], data_temp_2nd_filtered.x_test[2, :], color = :orange, legend=:bottomright, label="Test Data (2nd iteration)", legendfontsize = 10, tickfontsize = 10, labelfontsize = 14)
vline!([sampling_config_3rd.lb[1],sampling_config_3rd.ub[1]], label="x₁ bounds", linestyle=:dashdot, color=:red, linewidth = 2, legendfontsize = 10, tickfontsize = 10, labelfontsize = 14)
hline!([sampling_config_3rd.lb[2],sampling_config_3rd.ub[2]], label="x₂ bounds", linestyle=:dashdot, color=:purple, linewidth = 2, legendfontsize = 10, tickfontsize = 10, labelfontsize = 14)
savefig(joinpath(root, "images/exp1_3rd_scattered_point.pdf"))

println("# New added samples: ", sampling_config_3rd.n_samples)
println(" # Previous samples: " ,sampling_config_2nd_filtered.n_samples)
println(" # Filtered samples: ", size(data_temp_3rd_filtered.x_train)[2] + size(data_temp_3rd_filtered.x_test)[2])

sampling_config_3rd_filtered = Sampling_Config(
    sampling_config_3rd.n_samples,
    sampling_config_3rd.lb,
    sampling_config_3rd.ub
)

config_temp_3rd = NN_Config([6,50,50,1], [relu, relu, identity], false, 0.1, 0.3, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(0.1)), 1, size(data_temp_3rd_filtered.x_train)[2], 1000, 1)

train_time_3rd = @elapsed result_temp_3rd = NN_train(data_temp_3rd_filtered, config_temp_3rd, trained_model = model_2nd)
NN_results(config_temp_3rd, result_temp_3rd)

model_3rd = result_temp_3rd.model
BSON.@save joinpath(@__DIR__, "surrogate_3rd.bson") model_3rd
BSON.@load joinpath(@__DIR__, "surrogate_3rd.bson") model_3rd

# convert the surrogate model to a MILP model
MILP_bt_3rd = Model()
set_optimizer(MILP_bt_3rd, Gurobi.Optimizer)
set_silent(MILP_bt_3rd)
set_attribute(MILP_bt_3rd, "TimeLimit", 10)

build_time = @elapsed compressed_model_3rd, removed_neurons_3rd, bounds_U_3rd, bounds_L_3rd = NN_formulate!(MILP_bt_3rd, model_3rd, sampling_config_3rd_filtered.ub, sampling_config_3rd_filtered.lb; bound_tightening="fast", compress=true, silent=false)
@objective(MILP_bt_3rd, Min, MILP_bt_3rd[:x][3,1])

set_attribute(MILP_bt_3rd, "TimeLimit", 1800)
unset_silent(MILP_bt_3rd)
log_filename = "gurobi_log_3rd_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_3rd, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_3rd)
write_to_file(MILP_bt_3rd, joinpath(@__DIR__, "model_3rd_bt.mps"))

x_star_3rd = [value.(MILP_bt_3rd[:x][0,i]) for i in 1:length(MILP_bt_3rd[:x][0,:])]
solution_evaluate(MILP_bt_3rd, blade_max_temp)

# store multiple solutions in the solution pool
num_solutions_3rd_filtered = MOI.get(MILP_bt_3rd, MOI.ResultCount())
sol_pool_x_3rd_filtered, _ = sol_pool(MILP_bt_3rd, num_solutions_3rd_filtered)

# visualise the surrogate model
fig = plot_dual_contours(data_temp_3rd_filtered, model_3rd, x_star_3rd, "sub-optimal solutions", sol_pool_x_3rd_filtered, [1,2], 1)
Makie.save(joinpath(root, "images/exp1_3rd_dual_sol_pool.pdf"), fig)
