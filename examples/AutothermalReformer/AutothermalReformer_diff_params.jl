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
savefig(joinpath(root, "images/reparams/Scatter_plot_of_training_and_test_data.svg"))

# train the surrogate model
config_reformer = NN_Config([2,64,64,64,64,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.01, (0.9, 0.999), 1e-08), 1, 800, 1000, 0)
train_time = @elapsed result_reformer = NN_train(data_reformer_norm, config_reformer)
NN_results(config_reformer, result_reformer)
plot_learning_curve(config_reformer, result_reformer.err_hist)

model_init = result_reformer.model
BSON.@save joinpath(@__DIR__, "reparams/surrogate_init.bson") model_init
BSON.@load joinpath(@__DIR__, "reparams/surrogate_init.bson") model_init

# convert the surrogate model to a MILP model
@info "bound tightening and compression (Gogeta.jl)"
MILP_bt = Model()
set_optimizer(MILP_bt, Gurobi.Optimizer)
set_silent(MILP_bt)
set_attribute(MILP_bt, "TimeLimit", 10)
compressed_model, removed_neurons, bounds_U, bounds_L = NN_formulate!(MILP_bt, model_init, sampling_config_init_norm.ub, sampling_config_init_norm.lb; bound_tightening="standard", compress=true, silent=false);

# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_bt, Max, MILP_bt[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_bt, MILP_bt[:x][5,12] <= (0.34 - mean_init_y[12])/std_init_y[12])

set_attribute(MILP_bt, "TimeLimit", 1800)
unset_silent(MILP_bt)
log_filename = "gurobi_log_init_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt, "LogFile", joinpath(@__DIR__, "reparams/$log_filename"))
optimize!(MILP_bt)
write_to_file(MILP_bt, joinpath(@__DIR__, "reparams/model_init_bt.mps"))
MILP_bt = read_from_file(joinpath(@__DIR__, "reparams/model_init_bt.mps"))

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
Makie.save(joinpath(root, "images/reparams/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points.png"), fig)

#------------ apply Monte Carlo Dropout to the surrogate model ------------#
config_reformer_dp = NN_Config([2,64,64,64,64,12], [relu, relu, relu, relu, identity], false, 0, 0.1, Adam(0.001, (0.9, 0.999), 1e-07), 1, 800, 1000, 0)
train_time = @elapsed result_reformer = NN_train(data_reformer_norm, config_reformer_dp)
NN_results(config_reformer, result_reformer)
model_init_dp = result_reformer.model

pred, pred_dist, means, stds, x_top_std = predict_dist(data_reformer_norm, model_init_dp, 100, 10)
fig = plot_dual_contours(data_reformer_norm, model_init, x_star_init_norm, "x_top_std", [col for col in eachcol(x_top_std)], [1,2], 10)
Makie.save(joinpath(root, "images/reparams/Comparison of Simulator and Surrogate Model Contours with High Variance Points Marked.png"), fig)

# plot the predictive distribution of the 5th entry of x_test
pred_point, _ = predict_point(data_reformer_norm, model_init_dp, 100, 10)
println(data_reformer_norm.x_test[:, 5])
kdeplot(pred_dist[5][:,10], means[10,5])
savefig(joinpath(root, "images/Predictive Distribution of the 5th Entry of x_test.svg"))
# remove the outliers
pred_rm = remove_outliers_per_dist(pred_dist[5][:,10])
kdeplot(pred_rm, pred_point[5])

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
savefig(joinpath(root, "images/reparams/Scatter_plot_of_resampled_data.svg"))

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

config_reformer_1st = NN_Config([2,64,64,64,64,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 820, 1000, 1)
train_time_1st = @elapsed result_reformer_1st = NN_train(data_reformer_1st_filtered_norm, config_reformer_1st, trained_model = model_init)
NN_results(config_reformer_1st, result_reformer_1st)

model_1st = result_reformer_1st.model
BSON.@save joinpath(@__DIR__, "reparams/surrogate_1st.bson") model_1st
BSON.@load joinpath(@__DIR__, "reparams/surrogate_1st.bson") model_1st

# convert the surrogate model to a MILP model
MILP_bt_1st = Model()
set_optimizer(MILP_bt_1st, Gurobi.Optimizer)
set_silent(MILP_bt_1st)
set_attribute(MILP_bt_1st, "TimeLimit", 10)

compressed_model_1st, removed_neurons_1st, bounds_U_1st, bounds_L_1st = NN_formulate!(MILP_bt_1st, model_1st, sampling_config_1st_filtered_norm.ub, sampling_config_1st_filtered_norm.lb; bound_tightening="standard", compress=true, silent=false)

# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_bt_1st, Max, MILP_bt_1st[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_bt_1st, MILP_bt_1st[:x][5,12] <= (0.34 - mean_1st_filtered_y[12])/std_1st_filtered_y[12])

set_attribute(MILP_bt_1st, "TimeLimit", 1800)
unset_silent(MILP_bt_1st)
log_filename = "gurobi_log_1st_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_1st, "LogFile", joinpath(@__DIR__,  "reparams/$log_filename"))
optimize!(MILP_bt_1st)
write_to_file(MILP_bt_1st, joinpath(@__DIR__, "reparams/model_1st_bt.mps"))

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
Makie.save(joinpath(root, "images/reparams/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points 1st Iteration_filtered.png"), fig)

@info "option 2: train with the whole combined data"
size(data_reformer_1st.x_train)
size(data_reformer_1st.x_test)

data_reformer_1st_norm, mean_1st, std_1st, mean_1st_y, std_1st_y = normalise_data(data_reformer_1st, true)
sampling_config_1st_norm = Sampling_Config(
    sampling_config_1st.n_samples,
    (sampling_config_1st.lb .- vec(mean_1st)) ./ vec(std_1st),
    (sampling_config_1st.ub .- vec(mean_1st)) ./ vec(std_1st)
)

config_reformer_1st = NN_Config([2,64,64,64,64,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 866, 1000, 1)
train_time = @elapsed result_reformer_1st_whole = NN_train(data_reformer_1st_norm, config_reformer_1st, trained_model = model_init)
NN_results(config_reformer_1st, result_reformer_1st_whole)

model_1st_whole = result_reformer_1st_whole.model
BSON.@save joinpath(@__DIR__, "reparams/surrogate_1st_whole.bson") model_1st_whole
BSON.@load joinpath(@__DIR__, "reparams/surrogate_1st_whole.bson") model_1st_whole

# convert the surrogate model to a MILP model
MILP_bt_1st_whole = Model()
set_optimizer(MILP_bt_1st_whole, Gurobi.Optimizer)
set_silent(MILP_bt_1st_whole)
set_attribute(MILP_bt_1st_whole, "TimeLimit", 10)

compressed_model_1st_whole, removed_neurons_1st_whole, bounds_U_1st_whole, bounds_L_1st_whole = NN_formulate!(MILP_bt_1st_whole, model_1st_whole, sampling_config_1st_norm.ub, sampling_config_1st_norm.lb; bound_tightening="standard", compress=true, silent=false)

# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_bt_1st_whole, Max, MILP_bt_1st_whole[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_bt_1st_whole, MILP_bt_1st_whole[:x][5,12] <= (0.34 - mean_1st_y[12])/std_1st_y[12])

set_attribute(MILP_bt_1st_whole, "TimeLimit", 1800)
unset_silent(MILP_bt_1st_whole)
log_filename = "gurobi_log_1st_whole_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_1st_whole, "LogFile", joinpath(@__DIR__, "reparams/$log_filename"))
optimize!(MILP_bt_1st_whole)
write_to_file(MILP_bt_1st_whole, joinpath(@__DIR__, "reparams/model_1st_whole_bt.mps"))

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
Makie.save(joinpath(root, "images/reparams/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points 1st Iteration_whole.png"), fig)
