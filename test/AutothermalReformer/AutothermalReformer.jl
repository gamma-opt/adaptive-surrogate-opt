using Statistics
using Surrogates
using Flux
using JuMP
using Gurobi
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
    Auto-thermal Reformer Process (2 operating (or input) variables, 12 outputs of interest)
"""

Random.seed!(101)

#--------------------------------- Initial dataset ---------------------------------#
root = dirname(@__FILE__)
csv_file_path = joinpath(root, "data\\reformer.csv")
rdata = DataFrame(CSV.File(csv_file_path, header=true))
x = Matrix(rdata[:, 1:2])'
y = Matrix(rdata[:, 3:end-1])'
x_lb = minimum(x, dims=2)
x_ub = maximum(x, dims=2)
full_indices = 1:size(x, 2)

mean_total = mean(x, dims=2)
std_total = std(x, dims=2)
kdeplot(x[1,:], mean_total[1])
Plots.plot!(title = "Distribution of variable x₁", xlabel = "x₁", ylabel = "Density")
savefig(joinpath(root, "images/x1_density.svg"))
kdeplot(x[2,:], mean_total[2])
Plots.plot!(title = "Distribution of variable x₂", xlabel = "x₂", ylabel = "Density")
savefig(joinpath(root, "images/x2_density.svg"))

#--------------------------------- Initial training ---------------------------------#
# randomly select 1000 data points
selected_indices = randperm(size(x, 2))[1:1000]

# store the not selected data points for later use
complement_indices = setdiff(full_indices, selected_indices)
x_not_selected = x[:, complement_indices]
y_not_selected = y[:, complement_indices]

x_init = x[:, selected_indices]
y_init = y[:, selected_indices]
mean_init = mean(x_init, dims=2)
std_init = std(x_init, dims=2)
kdeplot(x_init[1,:], mean_init[1])
Plots.plot!(title = "Distribution of variable x₁", xlabel = "x₁", ylabel = "Density")
savefig(joinpath(root, "images/x1_density_selected.svg"))
kdeplot(x_init[2,:], mean_init[2])
Plots.plot!(title = "Distribution of variable x₂", xlabel = "x₂", ylabel = "Density")
savefig(joinpath(root, "images/x2_density_selected.svg"))

data_reformer = NN_Data()
train_data, test_data = Flux.splitobs((x_init, y_init), at = 0.8)
data_reformer.x_train = Float32.(train_data[1])
data_reformer.y_train = Float32.(train_data[2])
data_reformer.x_test = Float32.(test_data[1])
data_reformer.y_test = Float32.(test_data[2])

# define the lower and upper bounds for the input variables
L_bounds_init = [x_lb[1], x_lb[2]]
U_bounds_init = [x_ub[1], x_ub[2]]
sampling_config_init = Sampling_Config(1000, L_bounds_init, U_bounds_init)

# normalise the data, including the outputs
data_reformer_norm, mean_init, std_init, mean_init_y, std_init_y = normalise_data(data_reformer, true)

sampling_config_init_norm = Sampling_Config(
    sampling_config_init.n_samples,
    (sampling_config_init.lb .- vec(mean_init)) ./ vec(std_init),
    (sampling_config_init.ub .- vec(mean_init)) ./ vec(std_init)
)

Plots.scatter(data_reformer.x_train[1, :], data_reformer.x_train[2, :], color = :viridis, xlabel="x₁", ylabel="x₂", legend=:bottomright, label="Training Data")
Plots.scatter!(data_reformer.x_test[1, :], data_reformer.x_test[2, :], color = :orange, legend=:bottomright, label="Test Data")
savefig(joinpath(root, "images/Scatter_plot_of_training_and_test_data.svg"))

# train the surrogate model
config_reformer = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 800, 100, 0)
train_time = @elapsed result_reformer = NN_train(data_reformer_norm, config_reformer)
NN_results(config_reformer, result_reformer)
plot_learning_curve(config_reformer, result_reformer.err_hist)

model_init = result_reformer.model
BSON.@save joinpath(@__DIR__, "surrogate_init.bson") model_init
BSON.@load joinpath(@__DIR__, "surrogate_init.bson") model_init

#------------ big-M ------------#
# bounds of the input layer, and the other layers (arbitrary large big-M)
L_bounds = vcat(Float32.(sampling_config_init.lb), fill(Float32(-1e6), 52))
U_bounds = vcat(Float32.(sampling_config_init.ub), fill(Float32(1e6), 52))

# convert the trained nerual net to a JuMP model
build_time = @elapsed MILP_model = JuMP_Model(model_init, L_bounds, U_bounds)


# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_model, Max, MILP_model[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_model, MILP_model[:x][5,12] <= (0.34 - mean_init_y[12])/std_init_y[12])

optimize!(MILP_model)

#-------------------------------#

# convert the surrogate model to a MILP model
@info "bound tightening and compression (Gogeta.jl)"
MILP_bt = Model()
set_optimizer(MILP_bt, Gurobi.Optimizer)
set_silent(MILP_bt)
set_attribute(MILP_bt, "TimeLimit", 10)
build_time = @elapsed compressed_model, removed_neurons, bounds_U, bounds_L = NN_formulate!(MILP_bt, model_init, sampling_config_init_norm.ub, sampling_config_init_norm.lb; bound_tightening="standard", compress=true, silent=false);

# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_bt, Max, MILP_bt[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_bt, MILP_bt[:x][5,12] <= (0.34 - mean_init_y[12])/std_init_y[12])

set_attribute(MILP_bt, "TimeLimit", 1800)
unset_silent(MILP_bt)
log_filename = "gurobi_log_init_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt)
write_to_file(MILP_bt, joinpath(@__DIR__, "model_init_bt.mps"))
MILP_bt = read_from_file(joinpath(@__DIR__, "model_init_bt.mps"))

x_star_init_norm = [value.(MILP_bt[:x][0,i]) for i in 1:length(MILP_bt[:x][0,:])]
x_star_init = [value.(MILP_bt[:x][0,i]) for i in 1:length(MILP_bt[:x][0,:])] .* std_init .+ mean_init

println("Bypass Fraction:", x_star_init[1])
println("NG Steam Ratio:", x_star_init[2])
println("H2 Concentration:", objective_value(MILP_bt) * std_init_y[10] + mean_init_y[10])
println("N2 Concentration:", value.(MILP_bt[:x][5,12]) * std_init_y[12] + mean_init_y[12])

# store multiple solutions in the solution pool
num_solutions_init_bt = MOI.get(MILP_bt, MOI.ResultCount())
sol_pool_x_init_bt, _ = sol_pool(MILP_bt, num_solutions_init_bt, mean = mean_init, std = std_init)

# visualise the surrogate model 
fig = plot_dual_contours(data_reformer_norm, model_init, x_star_init_norm, "sol_pool", sol_pool_x_init_bt, [1,2], 10)
Makie.save(joinpath(root, "images/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points.png"), fig)

#------------ apply Monte Carlo Dropout to the surrogate model ------------#
config_reformer_dp = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.2, Adam(0.001, (0.9, 0.999), 1e-07), 1, 800, 100, 0)
train_time = @elapsed result_reformer = NN_train(data_reformer_norm, config_reformer_dp)
NN_results(config_reformer, result_reformer)
model_init_dp = result_reformer.model

pred, pred_dist, means, stds, x_top_std = predict_dist(data_reformer_norm, model_init_dp, 100, 10)
fig = plot_dual_contours(data_reformer_norm, model_init, x_star_init_norm, "x_top_std", [col for col in eachcol(x_top_std)], [1,2], 10)
Makie.save(joinpath(root, "images/Comparison of Simulator and Surrogate Model Contours with High Variance Points Marked.png"), fig)

# plot the predictive distribution of the 5th entry of x_test
pred_point, _ = predict_point(data_reformer_norm, model_init_dp, 100, 10)
println(data_reformer_norm.x_test[:, 5])
kdeplot(pred_dist[5][:,10], means[10,5])
savefig(joinpath(root, "images/Predictive Distribution of the 5th Entry of x_test.svg"))
# remove the outliers
pred_rm = remove_outliers_per_dist(pred_dist[5][:,10])
kdeplot(pred_rm, pred_point[5])

fig = plot_single_contour(data_reformer_norm, model_init_dp, x_star_init_norm, "Uncertainty of the Predictions", vec(stds),"x_top_std", [col for col in eachcol(x_top_std)], [1,2])
Makie.save(joinpath(root, "images/tricontour_Uncertainty of the Predictions_10.png"), fig)

#------------------------------ 1st iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_1st, sampling_config_1st = generate_resample_configs_mc(sampling_config_init_norm, [x_top_std x_star_init_norm], 0.10, 0.3, mean_init, std_init)
_, _, selected_indices_1st =  extract_data_from_given_dataset(x_not_selected, y_not_selected, sampling_configs_1st, complement_indices)

x_1st_added = x[:, selected_indices_1st]
y_1st_added = y[:, selected_indices_1st]
x_1st = x[:, vcat(selected_indices_1st, selected_indices)]
y_1st = y[:, vcat(selected_indices_1st, selected_indices)]

Plots.scatter(x_1st_added[1, :], x_1st_added[2, :], color = :lightgreen, xlabel="x₁", ylabel="x₂", legend=:bottomright, label="Resampled Data")
Plots.scatter!(data_reformer.x_train[1, :], data_reformer.x_train[2, :], color = :viridis, xlabel="x₁", ylabel="x₂", legend=:bottomright, label="Old Training Data")
Plots.scatter!(data_reformer.x_test[1, :], data_reformer.x_test[2, :], color = :orange, legend=:bottomright, label="Old Test Data")
vline!([sampling_config_1st.lb[1],sampling_config_1st.ub[1]], label="x1 bounds", linestyle=:dashdot, color=:red, linewidth = 2)
hline!([sampling_config_1st.lb[2],sampling_config_1st.ub[2]], label="x2 bounds", linestyle=:dashdot, color=:purple, linewidth = 2)
savefig(joinpath(root, "images/Scatter_plot_of_resampled_data.svg"))

data_reformer_1st = NN_Data()
train_data_1st, test_data_1st = Flux.splitobs((x_1st, y_1st), at = 0.8)
data_reformer_1st.x_train = Float32.(train_data_1st[1])
data_reformer_1st.y_train = Float32.(train_data_1st[2])
data_reformer_1st.x_test = Float32.(test_data_1st[1])
data_reformer_1st.y_test = Float32.(test_data_1st[2])

@info "option 1: train with the filtered combined data"
# train with the filtered combined data
data_reformer_1st_filtered = filter_data_within_bounds(data_reformer_1st, sampling_config_1st.lb, sampling_config_1st.ub)
size(data_reformer_1st_filtered.x_train)
size(data_reformer_1st_filtered.x_test)

data_reformer_1st_filtered_norm, mean_1st_filtered, std_1st_filtered, mean_1st_filtered_y, std_1st_filtered_y = normalise_data(data_reformer_1st_filtered, true)

sampling_config_1st_filtered_norm = Sampling_Config(
    sampling_config_1st.n_samples,
    (sampling_config_1st.lb .- vec(mean_1st_filtered)) ./ vec(std_1st_filtered),
    (sampling_config_1st.ub .- vec(mean_1st_filtered)) ./ vec(std_1st_filtered)
)

config_reformer_1st = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 223, 100, 1)
train_time_1st = @elapsed result_reformer_1st = NN_train(data_reformer_1st_filtered_norm, config_reformer_1st, trained_model = model_init)
NN_results(config_reformer_1st, result_reformer_1st)

model_1st = result_reformer_1st.model
BSON.@save joinpath(@__DIR__, "surrogate_1st.bson") model_1st
BSON.@load joinpath(@__DIR__, "surrogate_1st.bson") model_1st

# convert the surrogate model to a MILP model
MILP_bt_1st = Model()
set_optimizer(MILP_bt_1st, Gurobi.Optimizer)
set_silent(MILP_bt_1st)
set_attribute(MILP_bt_1st, "TimeLimit", 10)

build_time = @elapsed compressed_model_1st, removed_neurons_1st, bounds_U_1st, bounds_L_1st = NN_formulate!(MILP_bt_1st, model_1st, sampling_config_1st_filtered_norm.ub, sampling_config_1st_filtered_norm.lb; bound_tightening="standard", compress=true, silent=false)

# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_bt_1st, Max, MILP_bt_1st[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_bt_1st, MILP_bt_1st[:x][5,12] <= (0.34 - mean_1st_filtered_y[12])/std_1st_filtered_y[12])

set_attribute(MILP_bt_1st, "TimeLimit", 1800)
unset_silent(MILP_bt_1st)
log_filename = "gurobi_log_1st_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_1st, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_1st)
write_to_file(MILP_bt_1st, joinpath(@__DIR__, "model_1st_bt.mps"))

x_star_1st_norm = [value.(MILP_bt_1st[:x][0,i]) for i in 1:length(MILP_bt_1st[:x][0,:])]
x_star_1st = [value.(MILP_bt_1st[:x][0,i]) for i in 1:length(MILP_bt_1st[:x][0,:])] .* std_1st_filtered .+ mean_1st_filtered

println("Bypass Fraction:", x_star_1st[1])
println("NG Steam Ratio:", x_star_1st[2])
println("H2 Concentration:", objective_value(MILP_bt_1st) * std_1st_filtered_y[10] + mean_1st_filtered_y[10])
println("N2 Concentration:", value.(MILP_bt_1st[:x][5,12]) * std_1st_filtered_y[12] + mean_1st_filtered_y[12])

# store multiple solutions in the solution pool
num_solutions_1st_filtered = MOI.get(MILP_bt_1st, MOI.ResultCount())
sol_pool_x_1st_filtered, _ = sol_pool(MILP_bt_1st, num_solutions_1st_filtered, mean = mean_1st_filtered, std = std_1st_filtered)

# visualise the surrogate model
fig = plot_dual_contours(data_reformer_1st_filtered_norm, model_1st, x_star_1st_norm, "sol_pool", sol_pool_x_1st_filtered, [1,2], 10)
Makie.save(joinpath(root, "images/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points 1st Iteration_filtered.png"), fig)

@info "option 2: train with the whole combined data"
size(data_reformer_1st.x_train)
size(data_reformer_1st.x_test)

data_reformer_1st_norm, mean_1st, std_1st, mean_1st_y, std_1st_y = normalise_data(data_reformer_1st, true)
sampling_config_1st_norm = Sampling_Config(
    sampling_config_1st.n_samples,
    (sampling_config_1st.lb .- vec(mean_1st)) ./ vec(std_1st),
    (sampling_config_1st.ub .- vec(mean_1st)) ./ vec(std_1st)
)

config_reformer_1st = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 858, 100, 1)
train_time = @elapsed result_reformer_1st_whole = NN_train(data_reformer_1st_norm, config_reformer_1st, trained_model = model_init)
NN_results(config_reformer_1st, result_reformer_1st_whole)

model_1st_whole = result_reformer_1st_whole.model
BSON.@save joinpath(@__DIR__, "surrogate_1st_whole.bson") model_1st_whole
BSON.@load joinpath(@__DIR__, "surrogate_1st_whole.bson") model_1st_whole

# convert the surrogate model to a MILP model
MILP_bt_1st_whole = Model()
set_optimizer(MILP_bt_1st_whole, Gurobi.Optimizer)
set_silent(MILP_bt_1st_whole)
set_attribute(MILP_bt_1st_whole, "TimeLimit", 10)

# Layer 4 & 5 will be fully removed when setting "compress=true" and "bound_tightening=standard"
compressed_model_1st_whole, removed_neurons_1st_whole, bounds_U_1st_whole, bounds_L_1st_whole = NN_formulate!(MILP_bt_1st_whole, model_1st_whole, sampling_config_1st_norm.ub, sampling_config_1st_norm.lb; bound_tightening="fast", compress=true, silent=false)

# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_bt_1st_whole, Max, MILP_bt_1st_whole[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_bt_1st_whole, MILP_bt_1st_whole[:x][5,12] <= (0.34 - mean_1st_y[12])/std_1st_y[12])

set_attribute(MILP_bt_1st_whole, "TimeLimit", 1800)
unset_silent(MILP_bt_1st_whole)
log_filename = "gurobi_log_1st_whole_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_1st_whole, "LogFile", joinpath(@__DIR__, log_filename))
optimize!(MILP_bt_1st_whole)
write_to_file(MILP_bt_1st_whole, joinpath(@__DIR__, "model_1st_whole_bt.mps"))

x_star_1st_whole_norm = [value.(MILP_bt_1st_whole[:x][0,i]) for i in 1:length(MILP_bt_1st_whole[:x][0,:])]
x_star_1st_whole = [value.(MILP_bt_1st_whole[:x][0,i]) for i in 1:length(MILP_bt_1st_whole[:x][0,:])] .* std_1st .+ mean_1st

println("Bypass Fraction:", x_star_1st_whole[1])
println("NG Steam Ratio:", x_star_1st_whole[2])
println("H2 Concentration:", objective_value(MILP_bt_1st_whole) * std_1st_y[10] + mean_1st_y[10])
println("N2 Concentration:", value.(MILP_bt_1st_whole[:x][5,12]) * std_1st_y[12] + mean_1st_y[12])

# store multiple solutions in the solution pool
num_solutions_1st_whole = MOI.get(MILP_bt_1st_whole, MOI.ResultCount())
sol_pool_x_1st_whole, _ = sol_pool(MILP_bt_1st_whole, num_solutions_1st_whole, mean = mean_1st, std = std_1st)

# visualise the surrogate model
plot_dual_contours(data_reformer_1st_norm, model_1st_whole, x_star_1st_whole_norm, "sol_pool", sol_pool_x_1st_whole, [1,2], 10)
Makie.save(joinpath(root, "images/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points 1st Iteration_whole.png"), fig)

#------------------------------ 2nd iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_2nd, sampling_config_2nd = generate_resample_configs_mc(sampling_config_1st_filtered_norm, [x_top_std x_star_1st_norm], 0.10, 0.3, mean_1st_filtered, std_1st_filtered)
_, _, selected_indices_2nd =  extract_data_from_given_dataset(x_not_selected, y_not_selected, sampling_configs_2nd, complement_indices)

x_2nd_added = x[:, selected_indices_2nd]
y_2nd_added = y[:, selected_indices_2nd]
x_2nd = x[:, vcat(selected_indices_2nd, selected_indices_1st, selected_indices)]
y_2nd = y[:, vcat(selected_indices_2nd, selected_indices_1st, selected_indices)]

data_reformer_2nd = NN_Data()
train_data_2nd, test_data_2nd = Flux.splitobs((x_2nd, y_2nd), at = 0.8)
data_reformer_2nd.x_train = Float32.(train_data_2nd[1])
data_reformer_2nd.y_train = Float32.(train_data_2nd[2])
data_reformer_2nd.x_test = Float32.(test_data_2nd[1])
data_reformer_2nd.y_test = Float32.(test_data_2nd[2])

@info "option 1: train with the filtered combined data"
# train with the filtered combined data
data_reformer_2nd_filtered = filter_data_within_bounds(data_reformer_2nd, sampling_config_2nd.lb, sampling_config_2nd.ub)
size(data_reformer_2nd_filtered.x_train)
size(data_reformer_2nd_filtered.x_test)

data_reformer_2nd_filtered_norm, mean_2nd_filtered, std_2nd_filtered, mean_2nd_filtered_y, std_2nd_filtered_y = normalise_data(data_reformer_2nd_filtered, true)

sampling_config_2nd_filtered_norm = Sampling_Config(
    sampling_config_2nd.n_samples,
    (sampling_config_2nd.lb .- vec(mean_2nd_filtered)) ./ vec(std_2nd_filtered),
    (sampling_config_2nd.ub .- vec(mean_2nd_filtered)) ./ vec(std_2nd_filtered)
)

config_reformer_2nd = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 197, 100, 1)
train_time_2nd = @elapsed result_reformer_2nd = NN_train(data_reformer_2nd_filtered_norm, config_reformer_2nd, trained_model = model_1st)
NN_results(config_reformer_2nd, result_reformer_2nd)

model_2nd = result_reformer_2nd.model

# convert the surrogate model to a MILP model
MILP_bt_2nd = Model()
set_optimizer(MILP_bt_2nd, Gurobi.Optimizer)
set_silent(MILP_bt_2nd)
set_attribute(MILP_bt_2nd, "TimeLimit", 10)

build_time = @elapsed compressed_model_2nd, removed_neurons_2nd, bounds_U_2nd, bounds_L_2nd = NN_formulate!(MILP_bt_2nd, model_2nd, sampling_config_2nd_filtered_norm.ub, sampling_config_2nd_filtered_norm.lb; bound_tightening="standard", compress=true, silent=false)

# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_bt_2nd, Max, MILP_bt_2nd[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_bt_2nd, MILP_bt_2nd[:x][5,12] <= (0.34 - mean_2nd_filtered_y[12])/std_2nd_filtered_y[12])

set_attribute(MILP_bt_2nd, "TimeLimit", 1800)
unset_silent(MILP_bt_2nd)
# log_filename = "gurobi_log_2nd_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
# set_optimizer_attribute(MILP_bt_2nd, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_2nd)

x_star_2nd_norm = [value.(MILP_bt_2nd[:x][0,i]) for i in 1:length(MILP_bt_2nd[:x][0,:])]
x_star_2nd = [value.(MILP_bt_2nd[:x][0,i]) for i in 1:length(MILP_bt_2nd[:x][0,:])] .* std_2nd_filtered .+ mean_2nd_filtered

println("Bypass Fraction:", x_star_2nd[1])
println("NG Steam Ratio:", x_star_2nd[2])
println("H2 Concentration:", objective_value(MILP_bt_2nd) * std_2nd_filtered_y[10] + mean_2nd_filtered_y[10])
println("N2 Concentration:", value.(MILP_bt_2nd[:x][5,12]) * std_2nd_filtered_y[12] + mean_2nd_filtered_y[12])

fig = plot_dual_contours(data_reformer_2nd_filtered_norm, model_2nd, x_star_2nd_norm, "sol_pool", sol_pool_x_1st_filtered, [1,2], 10)
Makie.save(joinpath(root, "images/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points 2nd Iteration_filtered.png"), fig)

#------------------------------ 3rd iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_3rd, sampling_config_3rd = generate_resample_configs_mc(sampling_config_2nd_filtered_norm, [x_top_std x_star_2nd_norm], 0.10, 0.3, mean_2nd_filtered, std_2nd_filtered)
_, _, selected_indices_3rd =  extract_data_from_given_dataset(x_not_selected, y_not_selected, sampling_configs_3rd, complement_indices)

x_3rd_added = x[:, selected_indices_3rd]
y_3rd_added = y[:, selected_indices_3rd]
x_3rd = x[:, vcat(selected_indices_3rd, selected_indices_2nd, selected_indices_1st, selected_indices)]
y_3rd = y[:, vcat(selected_indices_3rd, selected_indices_2nd, selected_indices_1st, selected_indices)]

data_reformer_3rd = NN_Data()
train_data_3rd, test_data_3rd = Flux.splitobs((x_3rd, y_3rd), at = 0.8)
data_reformer_3rd.x_train = Float32.(train_data_3rd[1])
data_reformer_3rd.y_train = Float32.(train_data_3rd[2])
data_reformer_3rd.x_test = Float32.(test_data_3rd[1])
data_reformer_3rd.y_test = Float32.(test_data_3rd[2])

@info "option 1: train with the filtered combined data"
# train with the filtered combined data
data_reformer_3rd_filtered = filter_data_within_bounds(data_reformer_3rd, sampling_config_3rd.lb, sampling_config_3rd.ub)
size(data_reformer_3rd_filtered.x_train)
size(data_reformer_3rd_filtered.x_test)

data_reformer_3rd_filtered_norm, mean_3rd_filtered, std_3rd_filtered, mean_3rd_filtered_y, std_3rd_filtered_y = normalise_data(data_reformer_3rd_filtered, true)

sampling_config_3rd_filtered_norm = Sampling_Config(
    sampling_config_3rd.n_samples,
    (sampling_config_3rd.lb .- vec(mean_3rd_filtered)) ./ vec(std_3rd_filtered),
    (sampling_config_3rd.ub .- vec(mean_3rd_filtered)) ./ vec(std_3rd_filtered)
)

config_reformer_3rd = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 177, 100, 1)
train_time_3rd = @elapsed result_reformer_3rd = NN_train(data_reformer_3rd_filtered_norm, config_reformer_3rd, trained_model = model_2nd)
NN_results(config_reformer_3rd, result_reformer_3rd)

model_3rd = result_reformer_3rd.model

# convert the surrogate model to a MILP model

MILP_bt_3rd = Model()
set_optimizer(MILP_bt_3rd, Gurobi.Optimizer)
set_silent(MILP_bt_3rd)
set_attribute(MILP_bt_3rd, "TimeLimit", 10)

build_time = @elapsed compressed_model_3rd, removed_neurons_3rd, bounds_U_3rd, bounds_L_3rd = NN_formulate!(MILP_bt_3rd, model_3rd, sampling_config_3rd_filtered_norm.ub, sampling_config_3rd_filtered_norm.lb; bound_tightening="standard", compress=true, silent=false)

# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_bt_3rd, Max, MILP_bt_3rd[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_bt_3rd, MILP_bt_3rd[:x][5,12] <= (0.34 - mean_3rd_filtered_y[12])/std_3rd_filtered_y[12])

set_attribute(MILP_bt_3rd, "TimeLimit", 1800)
unset_silent(MILP_bt_3rd)
# log_filename = "gurobi_log_3rd_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
# set_optimizer_attribute(MILP_bt_3rd, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_3rd)

x_star_3rd_norm = [value.(MILP_bt_3rd[:x][0,i]) for i in 1:length(MILP_bt_3rd[:x][0,:])]
x_star_3rd = [value.(MILP_bt_3rd[:x][0,i]) for i in 1:length(MILP_bt_3rd[:x][0,:])] .* std_3rd_filtered .+ mean_3rd_filtered

println("Bypass Fraction:", x_star_3rd[1])
println("NG Steam Ratio:", x_star_3rd[2])
println("H2 Concentration:", objective_value(MILP_bt_3rd) * std_3rd_filtered_y[10] + mean_3rd_filtered_y[10])
println("N2 Concentration:", value.(MILP_bt_3rd[:x][5,12]) * std_3rd_filtered_y[12] + mean_3rd_filtered_y[12])

fig = plot_dual_contours(data_reformer_3rd_filtered_norm, model_3rd, x_star_3rd_norm, "sol_pool", sol_pool_x_1st_filtered, [1,2], 10)
Makie.save(joinpath(root, "images/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points 3rd Iteration_filtered.png"), fig)

#------------------------------ 4th iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_4th, sampling_config_4th = generate_resample_configs_mc(sampling_config_3rd_filtered_norm, [x_top_std x_star_3rd_norm], 0.10, 0.3, mean_3rd_filtered, std_3rd_filtered)
_, _, selected_indices_4th =  extract_data_from_given_dataset(x_not_selected, y_not_selected, sampling_configs_4th, complement_indices)

x_4th_added = x[:, selected_indices_4th]
y_4th_added = y[:, selected_indices_4th]
x_4th = x[:, vcat(selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st, selected_indices)]
y_4th = y[:, vcat(selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st, selected_indices)]

data_reformer_4th = NN_Data()
train_data_4th, test_data_4th = Flux.splitobs((x_4th, y_4th), at = 0.8)
data_reformer_4th.x_train = Float32.(train_data_4th[1])
data_reformer_4th.y_train = Float32.(train_data_4th[2])
data_reformer_4th.x_test = Float32.(test_data_4th[1])
data_reformer_4th.y_test = Float32.(test_data_4th[2])

@info "option 1: train with the filtered combined data"
# train with the filtered combined data
data_reformer_4th_filtered = filter_data_within_bounds(data_reformer_4th, sampling_config_4th.lb, sampling_config_4th.ub)
size(data_reformer_4th_filtered.x_train)
size(data_reformer_4th_filtered.x_test)

data_reformer_4th_filtered_norm, mean_4th_filtered, std_4th_filtered, mean_4th_filtered_y, std_4th_filtered_y = normalise_data(data_reformer_4th_filtered, true)

sampling_config_4th_filtered_norm = Sampling_Config(
    sampling_config_4th.n_samples,
    (sampling_config_4th.lb .- vec(mean_4th_filtered)) ./ vec(std_4th_filtered),
    (sampling_config_4th.ub .- vec(mean_4th_filtered)) ./ vec(std_4th_filtered)
)

config_reformer_4th = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 114, 100, 1)
train_time_4th = @elapsed result_reformer_4th = NN_train(data_reformer_4th_filtered_norm, config_reformer_4th, trained_model = model_3rd)
NN_results(config_reformer_4th, result_reformer_4th)

model_4th = result_reformer_4th.model

# convert the surrogate model to a MILP model

MILP_bt_4th = Model()
set_optimizer(MILP_bt_4th, Gurobi.Optimizer)
set_silent(MILP_bt_4th)
set_attribute(MILP_bt_4th, "TimeLimit", 10)

build_time = @elapsed compressed_model_4th, removed_neurons_4th, bounds_U_4th, bounds_L_4th = NN_formulate!(MILP_bt_4th, model_4th, sampling_config_4th_filtered_norm.ub, sampling_config_4th_filtered_norm.lb; bound_tightening="standard", compress=true, silent=false)

# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_bt_4th, Max, MILP_bt_4th[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_bt_4th, MILP_bt_4th[:x][5,12] <= (0.34 - mean_4th_filtered_y[12])/std_4th_filtered_y[12])

set_attribute(MILP_bt_4th, "TimeLimit", 1800)
unset_silent(MILP_bt_4th)
# log_filename = "gurobi_log_4th_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
# set_optimizer_attribute(MILP_bt_4th, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_4th)

x_star_4th_norm = [value.(MILP_bt_4th[:x][0,i]) for i in 1:length(MILP_bt_4th[:x][0,:])]
x_star_4th = [value.(MILP_bt_4th[:x][0,i]) for i in 1:length(MILP_bt_4th[:x][0,:])] .* std_4th_filtered .+ mean_4th_filtered

println("Bypass Fraction:", x_star_4th[1])
println("NG Steam Ratio:", x_star_4th[2])
println("H2 Concentration:", objective_value(MILP_bt_4th) * std_4th_filtered_y[10] + mean_4th_filtered_y[10])
println("N2 Concentration:", value.(MILP_bt_4th[:x][5,12]) * std_4th_filtered_y[12] + mean_4th_filtered_y[12])

num_solutions_4th_filtered = MOI.get(MILP_bt_4th, MOI.ResultCount())
sol_pool_x_4th_filtered, _ = sol_pool(MILP_bt_4th, num_solutions_4th_filtered, mean = mean_4th_filtered, std = std_4th_filtered)

fig = plot_dual_contours(data_reformer_4th_filtered_norm, model_4th, x_star_4th_norm, "sol_pool", sol_pool_x_4th_filtered, [1,2], 10)
Makie.save(joinpath(root, "images/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points 4th Iteration_filtered.png"), fig)


