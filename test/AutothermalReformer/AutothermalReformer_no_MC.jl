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
fig = plot_dual_contours(data_reformer_norm, model_init, x_star_init_norm, "sub-optimal solutions", sol_pool_x_init_bt, [1,2], 10)
Makie.save(joinpath(root, "images/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points.pdf"), fig)

#------------------------------ 1st iteration --------------------------#
# Resample densely around the points with the highest uncertainty
x_star_init_norm_m = reshape(x_star_init_norm, 2, 1)

sampling_configs_1st, _, resample_config_time_1st = generate_resample_configs_mc(sampling_config_init_norm, x_star_init_norm_m, 0.10, 0.3, mean_init, std_init)
_, _, selected_indices_1st, resample_data_time_1st =  extract_data_from_given_dataset(x_not_selected, y_not_selected, sampling_configs_1st, complement_indices)

x_1st_added = x[:, selected_indices_1st]
y_1st_added = y[:, selected_indices_1st]
x_1st = x[:, vcat(selected_indices_1st, selected_indices)]
y_1st = y[:, vcat(selected_indices_1st, selected_indices)]

Plots.scatter(x_1st_added[1, :], x_1st_added[2, :], color = :lightgreen, xlabel="x₁", ylabel="x₂", legend=:bottomright, label="Resampled Data", size = (530, 500), legendfontsize = 10, tickfontsize = 10, labelfontsize = 14)
Plots.scatter!(data_reformer.x_train[1, :], data_reformer.x_train[2, :], color = :viridis, xlabel="x₁", ylabel="x₂", legend=:bottomright, label="Training Data (initial) ", legendfontsize = 10, tickfontsize = 10, labelfontsize = 14)
Plots.scatter!(data_reformer.x_test[1, :], data_reformer.x_test[2, :], color = :orange, legend=:bottomright, label="Test Data (initial)", legendfontsize = 10, tickfontsize = 10, labelfontsize = 14)
vline!([sampling_config_1st.lb[1],sampling_config_1st.ub[1]], label="x₁ bounds", linestyle=:dashdot, color=:red, linewidth = 2, legendfontsize = 10, tickfontsize = 10, labelfontsize = 14)
hline!([sampling_config_1st.lb[2],sampling_config_1st.ub[2]], label="x₂ bounds", linestyle=:dashdot, color=:purple, linewidth = 2, legendfontsize = 10, tickfontsize = 10, labelfontsize = 14)
savefig(joinpath(root, "images/Scatter_plot_of_resampled_data.pdf"))

data_reformer_1st = NN_Data()
train_data_1st, test_data_1st = Flux.splitobs((x_1st, y_1st), at = 0.8)
data_reformer_1st.x_train = Float32.(train_data_1st[1])
data_reformer_1st.y_train = Float32.(train_data_1st[2])
data_reformer_1st.x_test = Float32.(test_data_1st[1])
data_reformer_1st.y_test = Float32.(test_data_1st[2])

@info "option 2: train with the whole combined data"
size(data_reformer_1st.x_train)
size(data_reformer_1st.x_test)

data_reformer_1st_norm, mean_1st, std_1st, mean_1st_y, std_1st_y = normalise_data(data_reformer_1st, true)
sampling_config_1st_norm = Sampling_Config(
    sampling_config_init.n_samples,
    (sampling_config_init.lb .- vec(mean_1st)) ./ vec(std_1st),
    (sampling_config_init.ub .- vec(mean_1st)) ./ vec(std_1st)
)

config_reformer_1st = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 803, 100, 1)
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
NN_formulate!(MILP_bt_1st_whole, model_1st_whole, sampling_config_1st_norm.ub, sampling_config_1st_norm.lb; bound_tightening="standard", compress=false, silent=false)

# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_bt_1st_whole, Max, MILP_bt_1st_whole[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_bt_1st_whole, MILP_bt_1st_whole[:x][5,12] <= (0.34 - mean_1st_y[12])/std_1st_y[12])

set_attribute(MILP_bt_1st_whole, "TimeLimit", 1800)
unset_silent(MILP_bt_1st_whole)
log_filename = "gurobi_log_1st_whole_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_1st_whole, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_1st_whole)
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
sampling_configs_2nd, _, resample_config_time_2nd = generate_resample_configs_mc(sampling_config_1st_norm, reshape(x_star_1st_whole_norm, 2, 1), 0.10, 0.3, mean_1st, std_1st)
_, _, selected_indices_2nd, resample_data_time_2nd =  extract_data_from_given_dataset(x_not_selected, y_not_selected, sampling_configs_2nd, complement_indices)

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

@info "option 2: train with the whole combined data"
size(data_reformer_2nd.x_train)
size(data_reformer_2nd.x_test)

data_reformer_2nd_norm, mean_2nd, std_2nd, mean_2nd_y, std_2nd_y = normalise_data(data_reformer_2nd, true)
sampling_config_2nd_norm = Sampling_Config(
    sampling_config_init.n_samples,
    (sampling_config_init.lb .- vec(mean_2nd)) ./ vec(std_2nd),
    (sampling_config_init.ub .- vec(mean_2nd)) ./ vec(std_2nd)
)

config_reformer_2nd = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 800, 100, 0)
train_time = @elapsed result_reformer_2nd_whole = NN_train(data_reformer_2nd_norm, config_reformer_2nd, trained_model = model_1st_whole)
NN_results(config_reformer_2nd, result_reformer_2nd_whole)

model_2nd_whole = result_reformer_2nd_whole.model
BSON.@save joinpath(@__DIR__, "surrogate_2nd_whole.bson") model_2nd_whole
BSON.@load joinpath(@__DIR__, "surrogate_2nd_whole.bson") model_2nd_whole

# convert the surrogate model to a MILP model
MILP_bt_2nd_whole = Model()
set_optimizer(MILP_bt_2nd_whole, Gurobi.Optimizer)
set_silent(MILP_bt_2nd_whole)
set_attribute(MILP_bt_2nd_whole, "TimeLimit", 10)

NN_formulate!(MILP_bt_2nd_whole, model_2nd_whole, sampling_config_2nd_norm.ub, sampling_config_2nd_norm.lb; bound_tightening="fast", compress=false, silent=false)

# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_bt_2nd_whole, Max, MILP_bt_2nd_whole[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_bt_2nd_whole, MILP_bt_2nd_whole[:x][5,12] <= (0.34 - mean_2nd_y[12])/std_2nd_y[12])

set_attribute(MILP_bt_2nd_whole, "TimeLimit", 1800)
unset_silent(MILP_bt_2nd_whole)
log_filename = "gurobi_log_2nd_whole_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_2nd_whole, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_2nd_whole)

x_star_2nd_whole_norm = [value.(MILP_bt_2nd_whole[:x][0,i]) for i in 1:length(MILP_bt_2nd_whole[:x][0,:])]
x_star_2nd_whole = [value.(MILP_bt_2nd_whole[:x][0,i]) for i in 1:length(MILP_bt_2nd_whole[:x][0,:])] .* std_2nd .+ mean_2nd

println("Bypass Fraction:", x_star_2nd_whole[1])
println("NG Steam Ratio:", x_star_2nd_whole[2])
println("H2 Concentration:", objective_value(MILP_bt_2nd_whole) * std_2nd_y[10] + mean_2nd_y[10])
println("N2 Concentration:", value.(MILP_bt_2nd_whole[:x][5,12]) * std_2nd_y[12] + mean_2nd_y[12])

fig = plot_dual_contours(data_reformer_2nd_norm, model_2nd_whole, x_star_2nd_whole_norm, "sol_pool", sol_pool_x_1st_filtered, [1,2], 10)
Makie.save(joinpath(root, "images/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points 2nd Iteration_whole.png"), fig)

#------------------------------ 3rd iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_3rd, _, resample_config_time_3rd = generate_resample_configs_mc(sampling_config_2nd_norm, reshape(x_star_2nd_whole_norm, 2, 1), 0.10, 0.3, mean_2nd, std_2nd)
_, _, selected_indices_3rd, resample_data_time_3rd =  extract_data_from_given_dataset(x_not_selected, y_not_selected, sampling_configs_3rd, complement_indices)

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

@info "option 2: train with the whole combined data"

data_reformer_3rd_norm, mean_3rd, std_3rd, mean_3rd_y, std_3rd_y = normalise_data(data_reformer_3rd, true)
sampling_config_3rd_norm = Sampling_Config(
    sampling_config_init.n_samples,
    (sampling_config_init.lb .- vec(mean_3rd)) ./ vec(std_3rd),
    (sampling_config_init.ub .- vec(mean_3rd)) ./ vec(std_3rd)
)

config_reformer_3rd = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 800, 100, 0)
train_time = @elapsed result_reformer_3rd_whole = NN_train(data_reformer_3rd_norm, config_reformer_3rd, trained_model = model_2nd_whole)
NN_results(config_reformer_3rd, result_reformer_3rd_whole)

model_3rd_whole = result_reformer_3rd_whole.model
BSON.@save joinpath(@__DIR__, "surrogate_3rd_whole.bson") model_3rd_whole
BSON.@load joinpath(@__DIR__, "surrogate_3rd_whole.bson") model_3rd_whole

# convert the surrogate model to a MILP model
MILP_bt_3rd_whole = Model()
set_optimizer(MILP_bt_3rd_whole, Gurobi.Optimizer)
set_silent(MILP_bt_3rd_whole)
set_attribute(MILP_bt_3rd_whole, "TimeLimit", 10)

NN_formulate!(MILP_bt_3rd_whole, model_3rd_whole, sampling_config_3rd_norm.ub, sampling_config_3rd_norm.lb; bound_tightening="standard", compress=false, silent=false)

# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_bt_3rd_whole, Max, MILP_bt_3rd_whole[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_bt_3rd_whole, MILP_bt_3rd_whole[:x][5,12] <= (0.34 - mean_3rd_y[12])/std_3rd_y[12])

set_attribute(MILP_bt_3rd_whole, "TimeLimit", 1800)
unset_silent(MILP_bt_3rd_whole)
log_filename = "gurobi_log_3rd_whole_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_3rd_whole, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_3rd_whole)

x_star_3rd_whole_norm = [value.(MILP_bt_3rd_whole[:x][0,i]) for i in 1:length(MILP_bt_3rd_whole[:x][0,:])]
x_star_3rd_whole = [value.(MILP_bt_3rd_whole[:x][0,i]) for i in 1:length(MILP_bt_3rd_whole[:x][0,:])] .* std_3rd .+ mean_3rd

println("Bypass Fraction:", x_star_3rd_whole[1])
println("NG Steam Ratio:", x_star_3rd_whole[2])
println("H2 Concentration:", objective_value(MILP_bt_3rd_whole) * std_3rd_y[10] + mean_3rd_y[10])
println("N2 Concentration:", value.(MILP_bt_3rd_whole[:x][5,12]) * std_3rd_y[12] + mean_3rd_y[12])

#------------------------------ 4th iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_4th, _ = generate_resample_configs_mc(sampling_config_3rd_norm, reshape(x_star_3rd_whole_norm, 2, 1), 0.10, 0.3, mean_3rd, std_3rd)
_, _, selected_indices_4th, resample_data_time_4th =  extract_data_from_given_dataset(x_not_selected, y_not_selected, sampling_configs_4th, complement_indices)

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

@info "option 2: train with the whole combined data"

data_reformer_4th_norm, mean_4th, std_4th, mean_4th_y, std_4th_y = normalise_data(data_reformer_4th, true)
sampling_config_4th_norm = Sampling_Config(
    sampling_config_init.n_samples,
    (sampling_config_init.lb .- vec(mean_4th)) ./ vec(std_4th),
    (sampling_config_init.ub .- vec(mean_4th)) ./ vec(std_4th)
)

config_reformer_4th = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 800, 100, 0)
train_time = @elapsed result_reformer_4th_whole = NN_train(data_reformer_4th_norm, config_reformer_4th, trained_model = model_3rd_whole)
NN_results(config_reformer_4th, result_reformer_4th_whole)

model_4th_whole = result_reformer_4th_whole.model
BSON.@save joinpath(@__DIR__, "surrogate_4th_whole.bson") model_4th_whole
BSON.@load joinpath(@__DIR__, "surrogate_4th_whole.bson") model_4th_whole

# convert the surrogate model to a MILP model
MILP_bt_4th_whole = Model()
set_optimizer(MILP_bt_4th_whole, Gurobi.Optimizer)
set_silent(MILP_bt_4th_whole)
set_attribute(MILP_bt_4th_whole, "TimeLimit", 10)

NN_formulate!(MILP_bt_4th_whole, model_4th_whole, sampling_config_4th_norm.ub, sampling_config_4th_norm.lb; bound_tightening="standard", compress=false, silent=false)

# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_bt_4th_whole, Max, MILP_bt_4th_whole[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_bt_4th_whole, MILP_bt_4th_whole[:x][5,12] <= (0.34 - mean_4th_y[12])/std_4th_y[12])

set_attribute(MILP_bt_4th_whole, "TimeLimit", 1800)
unset_silent(MILP_bt_4th_whole)
log_filename = "gurobi_log_4th_whole_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_4th_whole, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_4th_whole)

x_star_4th_whole_norm = [value.(MILP_bt_4th_whole[:x][0,i]) for i in 1:length(MILP_bt_4th_whole[:x][0,:])]
x_star_4th_whole = [value.(MILP_bt_4th_whole[:x][0,i]) for i in 1:length(MILP_bt_4th_whole[:x][0,:])] .* std_4th .+ mean_4th

println("Bypass Fraction:", x_star_4th_whole[1])
println("NG Steam Ratio:", x_star_4th_whole[2])
println("H2 Concentration:", objective_value(MILP_bt_4th_whole) * std_4th_y[10] + mean_4th_y[10])
println("N2 Concentration:", value.(MILP_bt_4th_whole[:x][5,12]) * std_4th_y[12] + mean_4th_y[12])

fig = plot_dual_contours(data_reformer_4th_norm, model_4th_whole, x_star_4th_whole_norm, "sol_pool", sol_pool_x_1st_filtered, [1,2], 10)
Makie.save(joinpath(root, "images/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points 4th Iteration_whole.png"), fig)

#------------------------------ 5th iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_5th, _ = generate_resample_configs_mc(sampling_config_4th_norm, reshape(x_star_4th_whole_norm, 2, 1), 0.10, 0.3, mean_4th, std_4th)
_, _, selected_indices_5th, resample_data_time_5th =  extract_data_from_given_dataset(x_not_selected, y_not_selected, sampling_configs_5th, complement_indices)

x_5th_added = x[:, selected_indices_5th]
y_5th_added = y[:, selected_indices_5th]
x_5th = x[:, vcat(selected_indices_5th, selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st, selected_indices)]
y_5th = y[:, vcat(selected_indices_5th, selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st, selected_indices)]

data_reformer_5th = NN_Data()
train_data_5th, test_data_5th = Flux.splitobs((x_5th, y_5th), at = 0.8)
data_reformer_5th.x_train = Float32.(train_data_5th[1])
data_reformer_5th.y_train = Float32.(train_data_5th[2])
data_reformer_5th.x_test = Float32.(test_data_5th[1])
data_reformer_5th.y_test = Float32.(test_data_5th[2])

@info "option 2: train with the whole combined data"

data_reformer_5th_norm, mean_5th, std_5th, mean_5th_y, std_5th_y = normalise_data(data_reformer_5th, true)
sampling_config_5th_norm = Sampling_Config(
    sampling_config_init.n_samples,
    (sampling_config_init.lb .- vec(mean_5th)) ./ vec(std_5th),
    (sampling_config_init.ub .- vec(mean_5th)) ./ vec(std_5th)
)

config_reformer_5th = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 800, 100, 0)
train_time = @elapsed result_reformer_5th_whole = NN_train(data_reformer_5th_norm, config_reformer_5th, trained_model = model_4th_whole)
NN_results(config_reformer_5th, result_reformer_5th_whole)

model_5th_whole = result_reformer_5th_whole.model
BSON.@save joinpath(@__DIR__, "surrogate_5th_whole.bson") model_5th_whole
BSON.@load joinpath(@__DIR__, "surrogate_5th_whole.bson") model_5th_whole

# convert the surrogate model to a MILP model
MILP_bt_5th_whole = Model()
set_optimizer(MILP_bt_5th_whole, Gurobi.Optimizer)
set_silent(MILP_bt_5th_whole)
set_attribute(MILP_bt_5th_whole, "TimeLimit", 10)

NN_formulate!(MILP_bt_5th_whole, model_5th_whole, sampling_config_5th_norm.ub, sampling_config_5th_norm.lb; bound_tightening="standard", compress=false, silent=false)

# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_bt_5th_whole, Max, MILP_bt_5th_whole[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_bt_5th_whole, MILP_bt_5th_whole[:x][5,12] <= (0.34 - mean_5th_y[12])/std_5th_y[12])

set_attribute(MILP_bt_5th_whole, "TimeLimit", 1800)
unset_silent(MILP_bt_5th_whole)
log_filename = "gurobi_log_5th_whole_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_5th_whole, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_5th_whole)

x_star_5th_whole_norm = [value.(MILP_bt_5th_whole[:x][0,i]) for i in 1:length(MILP_bt_5th_whole[:x][0,:])]
x_star_5th_whole = [value.(MILP_bt_5th_whole[:x][0,i]) for i in 1:length(MILP_bt_5th_whole[:x][0,:])] .* std_5th .+ mean_5th

println("Bypass Fraction:", x_star_5th_whole[1])
println("NG Steam Ratio:", x_star_5th_whole[2])
println("H2 Concentration:", objective_value(MILP_bt_5th_whole) * std_5th_y[10] + mean_5th_y[10])
println("N2 Concentration:", value.(MILP_bt_5th_whole[:x][5,12]) * std_5th_y[12] + mean_5th_y[12])

fig = plot_dual_contours(data_reformer_5th_norm, model_5th_whole, x_star_5th_whole_norm, "sol_pool", sol_pool_x_1st_filtered, [1,2], 10)
Makie.save(joinpath(root, "images/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points 5th Iteration_whole.png"), fig)

#------------------------------ 6th iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_6th, _ = generate_resample_configs_mc(sampling_config_5th_norm, reshape(x_star_5th_whole_norm, 2, 1), 0.10, 0.3, mean_5th, std_5th)
_, _, selected_indices_6th, resample_data_time_6th =  extract_data_from_given_dataset(x_not_selected, y_not_selected, sampling_configs_6th, complement_indices)

x_6th_added = x[:, selected_indices_6th]
y_6th_added = y[:, selected_indices_6th]
x_6th = x[:, vcat(selected_indices_6th, selected_indices_5th, selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st, selected_indices)]
y_6th = y[:, vcat(selected_indices_6th, selected_indices_5th, selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st, selected_indices)]

data_reformer_6th = NN_Data()
train_data_6th, test_data_6th = Flux.splitobs((x_6th, y_6th), at = 0.8)
data_reformer_6th.x_train = Float32.(train_data_6th[1])
data_reformer_6th.y_train = Float32.(train_data_6th[2])
data_reformer_6th.x_test = Float32.(test_data_6th[1])
data_reformer_6th.y_test = Float32.(test_data_6th[2])

@info "option 2: train with the whole combined data"

data_reformer_6th_norm, mean_6th, std_6th, mean_6th_y, std_6th_y = normalise_data(data_reformer_6th, true)
sampling_config_6th_norm = Sampling_Config(
    sampling_config_init.n_samples,
    (sampling_config_init.lb .- vec(mean_6th)) ./ vec(std_6th),
    (sampling_config_init.ub .- vec(mean_6th)) ./ vec(std_6th)
)

config_reformer_6th = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 800, 100, 0)
train_time = @elapsed result_reformer_6th_whole = NN_train(data_reformer_6th_norm, config_reformer_6th, trained_model = model_5th_whole)
NN_results(config_reformer_6th, result_reformer_6th_whole)

model_6th_whole = result_reformer_6th_whole.model
BSON.@save joinpath(@__DIR__, "surrogate_6th_whole.bson") model_6th_whole
BSON.@load joinpath(@__DIR__, "surrogate_6th_whole.bson") model_6th_whole

# convert the surrogate model to a MILP model
MILP_bt_6th_whole = Model()
set_optimizer(MILP_bt_6th_whole, Gurobi.Optimizer)
set_silent(MILP_bt_6th_whole)
set_attribute(MILP_bt_6th_whole, "TimeLimit", 10)

NN_formulate!(MILP_bt_6th_whole, model_6th_whole, sampling_config_6th_norm.ub, sampling_config_6th_norm.lb; bound_tightening="standard", compress=false, silent=false)

# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_bt_6th_whole, Max, MILP_bt_6th_whole[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_bt_6th_whole, MILP_bt_6th_whole[:x][5,12] <= (0.34 - mean_6th_y[12])/std_6th_y[12])

set_attribute(MILP_bt_6th_whole, "TimeLimit", 1800)
unset_silent(MILP_bt_6th_whole)
log_filename = "gurobi_log_6th_whole_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_6th_whole, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_6th_whole)

x_star_6th_whole_norm = [value.(MILP_bt_6th_whole[:x][0,i]) for i in 1:length(MILP_bt_6th_whole[:x][0,:])]
x_star_6th_whole = [value.(MILP_bt_6th_whole[:x][0,i]) for i in 1:length(MILP_bt_6th_whole[:x][0,:])] .* std_6th .+ mean_6th

println("Bypass Fraction:", x_star_6th_whole[1])
println("NG Steam Ratio:", x_star_6th_whole[2])
println("H2 Concentration:", objective_value(MILP_bt_6th_whole) * std_6th_y[10] + mean_6th_y[10])
println("N2 Concentration:", value.(MILP_bt_6th_whole[:x][5,12]) * std_6th_y[12] + mean_6th_y[12])

fig = plot_dual_contours(data_reformer_6th_norm, model_6th_whole, x_star_6th_whole_norm, "sol_pool", sol_pool_x_1st_filtered, [1,2], 10)
Makie.save(joinpath(root, "images/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points 6th Iteration_whole.png"), fig)

#------------------------------ 7th iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_7th, _ = generate_resample_configs_mc(sampling_config_6th_norm, reshape(x_star_6th_whole_norm, 2, 1), 0.10, 0.3, mean_6th, std_6th)
_, _, selected_indices_7th, resample_data_time_7th =  extract_data_from_given_dataset(x_not_selected, y_not_selected, sampling_configs_7th, complement_indices)

x_7th_added = x[:, selected_indices_7th]
y_7th_added = y[:, selected_indices_7th]
x_7th = x[:, vcat(selected_indices_7th, selected_indices_6th, selected_indices_5th, selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st, selected_indices)]
y_7th = y[:, vcat(selected_indices_7th, selected_indices_6th, selected_indices_5th, selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st, selected_indices)]

data_reformer_7th = NN_Data()
train_data_7th, test_data_7th = Flux.splitobs((x_7th, y_7th), at = 0.8)
data_reformer_7th.x_train = Float32.(train_data_7th[1])
data_reformer_7th.y_train = Float32.(train_data_7th[2])
data_reformer_7th.x_test = Float32.(test_data_7th[1])
data_reformer_7th.y_test = Float32.(test_data_7th[2])

@info "option 2: train with the whole combined data"

data_reformer_7th_norm, mean_7th, std_7th, mean_7th_y, std_7th_y = normalise_data(data_reformer_7th, true)
sampling_config_7th_norm = Sampling_Config(
    sampling_config_init.n_samples,
    (sampling_config_init.lb .- vec(mean_7th)) ./ vec(std_7th),
    (sampling_config_init.ub .- vec(mean_7th)) ./ vec(std_7th)
)

config_reformer_7th = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 800, 100, 0)
train_time = @elapsed result_reformer_7th_whole = NN_train(data_reformer_7th_norm, config_reformer_7th, trained_model = model_6th_whole)
NN_results(config_reformer_7th, result_reformer_7th_whole)

model_7th_whole = result_reformer_7th_whole.model
BSON.@save joinpath(@__DIR__, "surrogate_7th_whole.bson") model_7th_whole
BSON.@load joinpath(@__DIR__, "surrogate_7th_whole.bson") model_7th_whole

# convert the surrogate model to a MILP model
MILP_bt_7th_whole = Model()
set_optimizer(MILP_bt_7th_whole, Gurobi.Optimizer)
set_silent(MILP_bt_7th_whole)
set_attribute(MILP_bt_7th_whole, "TimeLimit", 10)

NN_formulate!(MILP_bt_7th_whole, model_7th_whole, sampling_config_7th_norm.ub, sampling_config_7th_norm.lb; bound_tightening="standard", compress=false, silent=false)

# m.obj = pyo.Objective(expr=m.reformer.outputs[h2_idx], sense=pyo.maximize)
@objective(MILP_bt_7th_whole, Max, MILP_bt_7th_whole[:x][5,10])
# m.con = pyo.Constraint(expr=m.reformer.outputs[n2_idx] <= 0.34)
@constraint(MILP_bt_7th_whole, MILP_bt_7th_whole[:x][5,12] <= (0.34 - mean_7th_y[12])/std_7th_y[12])

set_attribute(MILP_bt_7th_whole, "TimeLimit", 1800)
unset_silent(MILP_bt_7th_whole)
log_filename = "gurobi_log_7th_whole_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_7th_whole, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_7th_whole)

x_star_7th_whole_norm = [value.(MILP_bt_7th_whole[:x][0,i]) for i in 1:length(MILP_bt_7th_whole[:x][0,:])]
x_star_7th_whole = [value.(MILP_bt_7th_whole[:x][0,i]) for i in 1:length(MILP_bt_7th_whole[:x][0,:])] .* std_7th .+ mean_7th

println("Bypass Fraction:", x_star_7th_whole[1])
println("NG Steam Ratio:", x_star_7th_whole[2])
println("H2 Concentration:", objective_value(MILP_bt_7th_whole) * std_7th_y[10] + mean_7th_y[10])
println("N2 Concentration:", value.(MILP_bt_7th_whole[:x][5,12]) * std_7th_y[12] + mean_7th_y[12])

fig = plot_dual_contours(data_reformer_7th_norm, model_7th_whole, x_star_7th_whole_norm, "sol_pool", sol_pool_x_1st_filtered, [1,2], 10)
Makie.save(joinpath(root, "images/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points 7th Iteration_whole.png"), fig)

#------------------------------ 8th iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_8th, _ = generate_resample_configs_mc(sampling_config_7th_norm, reshape(x_star_7th_whole_norm, 2, 1), 0.10, 0.3, mean_7th, std_7th)
_, _, selected_indices_8th, resample_data_time_8th =  extract_data_from_given_dataset(x_not_selected, y_not_selected, sampling_configs_8th, complement_indices)

x_8th_added = x[:, selected_indices_8th]
y_8th_added = y[:, selected_indices_8th]
x_8th = x[:, vcat(selected_indices_8th, selected_indices_7th, selected_indices_6th, selected_indices_5th, selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st, selected_indices)]
y_8th = y[:, vcat(selected_indices_8th, selected_indices_7th, selected_indices_6th, selected_indices_5th, selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st, selected_indices)]

data_reformer_8th = NN_Data()
train_data_8th, test_data_8th = Flux.splitobs((x_8th, y_8th), at = 0.8)
data_reformer_8th.x_train = Float32.(train_data_8th[1])
data_reformer_8th.y_train = Float32.(train_data_8th[2])
data_reformer_8th.x_test = Float32.(test_data_8th[1])
data_reformer_8th.y_test = Float32.(test_data_8th[2])

@info "option 2: train with the whole combined data"

data_reformer_8th_norm, mean_8th, std_8th, mean_8th_y, std_8th_y = normalise_data(data_reformer_8th, true)
sampling_config_8th_norm = Sampling_Config(
    sampling_config_init.n_samples,
    (sampling_config_init.lb .- vec(mean_8th)) ./ vec(std_8th),
    (sampling_config_init.ub .- vec(mean_8th)) ./ vec(std_8th)
)
config_reformer_8th = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 800, 100, 0)
train_time = @elapsed result_reformer_8th_whole = NN_train(data_reformer_8th_norm, config_reformer_8th, trained_model = model_7th_whole)
NN_results(config_reformer_8th, result_reformer_8th_whole)

model_8th_whole = result_reformer_8th_whole.model
BSON.@save joinpath(@__DIR__, "surrogate_8th_whole.bson") model_8th_whole
BSON.@load joinpath(@__DIR__, "surrogate_8th_whole.bson") model_8th_whole

# convert the surrogate model to a MILP model
MILP_bt_8th_whole = Model()
set_optimizer(MILP_bt_8th_whole, Gurobi.Optimizer)
set_silent(MILP_bt_8th_whole)
set_attribute(MILP_bt_8th_whole, "TimeLimit", 10)

NN_formulate!(MILP_bt_8th_whole, model_8th_whole, sampling_config_8th_norm.ub, sampling_config_8th_norm.lb; bound_tightening="standard", compress=false, silent=false)

@objective(MILP_bt_8th_whole, Max, MILP_bt_8th_whole[:x][5,10])
@constraint(MILP_bt_8th_whole, MILP_bt_8th_whole[:x][5,12] <= (0.34 - mean_8th_y[12])/std_8th_y[12])

set_attribute(MILP_bt_8th_whole, "TimeLimit", 1800)
unset_silent(MILP_bt_8th_whole)
log_filename = "gurobi_log_8th_whole_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_8th_whole, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_8th_whole)

x_star_8th_whole_norm = [value.(MILP_bt_8th_whole[:x][0,i]) for i in 1:length(MILP_bt_8th_whole[:x][0,:])]
x_star_8th_whole = [value.(MILP_bt_8th_whole[:x][0,i]) for i in 1:length(MILP_bt_8th_whole[:x][0,:])] .* std_8th .+ mean_8th

println("Bypass Fraction:", x_star_8th_whole[1])
println("NG Steam Ratio:", x_star_8th_whole[2])
println("H2 Concentration:", objective_value(MILP_bt_8th_whole) * std_8th_y[10] + mean_8th_y[10])
println("N2 Concentration:", value.(MILP_bt_8th_whole[:x][5,12]) * std_8th_y[12] + mean_8th_y[12])


#------------------------------ 9th iteration --------------------------#
sampling_configs_9th, _ = generate_resample_configs_mc(sampling_config_8th_norm, reshape(x_star_8th_whole_norm, 2, 1), 0.10, 0.3, mean_8th, std_8th)
_, _, selected_indices_9th, resample_data_time_9th = extract_data_from_given_dataset(x_not_selected, y_not_selected, sampling_configs_9th, complement_indices)

x_9th_added = x[:, selected_indices_9th]
y_9th_added = y[:, selected_indices_9th]
x_9th = x[:, vcat(selected_indices_9th, selected_indices_8th, selected_indices_7th, selected_indices_6th, selected_indices_5th, selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st, selected_indices)]
y_9th = y[:, vcat(selected_indices_9th, selected_indices_8th, selected_indices_7th, selected_indices_6th, selected_indices_5th, selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st, selected_indices)]

data_reformer_9th = NN_Data()
train_data_9th, test_data_9th = Flux.splitobs((x_9th, y_9th), at = 0.8)
data_reformer_9th.x_train = Float32.(train_data_9th[1])
data_reformer_9th.y_train = Float32.(train_data_9th[2])
data_reformer_9th.x_test = Float32.(test_data_9th[1])
data_reformer_9th.y_test = Float32.(test_data_9th[2])

@info "option 2: train with the whole combined data"

data_reformer_9th_norm, mean_9th, std_9th, mean_9th_y, std_9th_y = normalise_data(data_reformer_9th, true)
sampling_config_9th_norm = Sampling_Config(
    sampling_config_init.n_samples,
    (sampling_config_init.lb .- vec(mean_9th)) ./ vec(std_9th),
    (sampling_config_init.ub .- vec(mean_9th)) ./ vec(std_9th)
)

config_reformer_9th = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 800, 100, 0)
train_time = @elapsed result_reformer_9th_whole = NN_train(data_reformer_9th_norm, config_reformer_9th, trained_model = model_8th_whole)
NN_results(config_reformer_9th, result_reformer_9th_whole)

model_9th_whole = result_reformer_9th_whole.model
BSON.@save joinpath(@__DIR__, "surrogate_9th_whole.bson") model_9th_whole
BSON.@load joinpath(@__DIR__, "surrogate_9th_whole.bson") model_9th_whole

MILP_bt_9th_whole = Model()
set_optimizer(MILP_bt_9th_whole, Gurobi.Optimizer)
set_silent(MILP_bt_9th_whole)
set_attribute(MILP_bt_9th_whole, "TimeLimit", 10)

NN_formulate!(MILP_bt_9th_whole, model_9th_whole, sampling_config_9th_norm.ub, sampling_config_9th_norm.lb; bound_tightening="standard", compress=false, silent=false)

@objective(MILP_bt_9th_whole, Max, MILP_bt_9th_whole[:x][5,10])
@constraint(MILP_bt_9th_whole, MILP_bt_9th_whole[:x][5,12] <= (0.34 - mean_9th_y[12])/std_9th_y[12])

set_attribute(MILP_bt_9th_whole, "TimeLimit", 1800)
unset_silent(MILP_bt_9th_whole)
log_filename = "gurobi_log_9th_whole_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_9th_whole, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_9th_whole)

x_star_9th_whole_norm = [value.(MILP_bt_9th_whole[:x][0,i]) for i in 1:length(MILP_bt_9th_whole[:x][0,:])]
x_star_9th_whole = [value.(MILP_bt_9th_whole[:x][0,i]) for i in 1:length(MILP_bt_9th_whole[:x][0,:])] .* std_9th .+ mean_9th

println("Bypass Fraction:", x_star_9th_whole[1])
println("NG Steam Ratio:", x_star_9th_whole[2])
println("H2 Concentration:", objective_value(MILP_bt_9th_whole) * std_9th_y[10] + mean_9th_y[10])
println("N2 Concentration:", value.(MILP_bt_9th_whole[:x][5,12]) * std_9th_y[12] + mean_9th_y[12])

fig = plot_dual_contours(data_reformer_9th_norm, model_9th_whole, x_star_9th_whole_norm, "sol_pool", sol_pool_x_1st_filtered, [1,2], 10)
Makie.save(joinpath(root, "images/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points 9th Iteration_whole.png"), fig)

#------------------------------ 10th iteration --------------------------#
sampling_configs_10th, _ = generate_resample_configs_mc(sampling_config_9th_norm, reshape(x_star_9th_whole_norm, 2, 1), 0.10, 0.3, mean_9th, std_9th)
_, _, selected_indices_10th, resample_data_time_10th = extract_data_from_given_dataset(x_not_selected, y_not_selected, sampling_configs_10th, complement_indices)

x_10th_added = x[:, selected_indices_10th]
y_10th_added = y[:, selected_indices_10th]
x_10th = x[:, vcat(selected_indices_10th, selected_indices_9th, selected_indices_8th, selected_indices_7th, selected_indices_6th, selected_indices_5th, selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st, selected_indices)]
y_10th = y[:, vcat(selected_indices_10th, selected_indices_9th, selected_indices_8th, selected_indices_7th, selected_indices_6th, selected_indices_5th, selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st, selected_indices)]

data_reformer_10th = NN_Data()
train_data_10th, test_data_10th = Flux.splitobs((x_10th, y_10th), at = 0.8)
data_reformer_10th.x_train = Float32.(train_data_10th[1])
data_reformer_10th.y_train = Float32.(train_data_10th[2])
data_reformer_10th.x_test = Float32.(test_data_10th[1])
data_reformer_10th.y_test = Float32.(test_data_10th[2])

@info "option 2: train with the whole combined data"

data_reformer_10th_norm, mean_10th, std_10th, mean_10th_y, std_10th_y = normalise_data(data_reformer_10th, true)
sampling_config_10th_norm = Sampling_Config(
    sampling_config_init.n_samples,
    (sampling_config_init.lb .- vec(mean_10th)) ./ vec(std_10th),
    (sampling_config_init.ub .- vec(mean_10th)) ./ vec(std_10th)
)

config_reformer_10th = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 800, 100, 0)
train_time = @elapsed result_reformer_10th_whole = NN_train(data_reformer_10th_norm, config_reformer_10th, trained_model = model_9th_whole)
NN_results(config_reformer_10th, result_reformer_10th_whole)

model_10th_whole = result_reformer_10th_whole.model
BSON.@save joinpath(@__DIR__, "surrogate_10th_whole.bson") model_10th_whole
BSON.@load joinpath(@__DIR__, "surrogate_10th_whole.bson") model_10th_whole

MILP_bt_10th_whole = Model()
set_optimizer(MILP_bt_10th_whole, Gurobi.Optimizer)
set_silent(MILP_bt_10th_whole)
set_attribute(MILP_bt_10th_whole, "TimeLimit", 10)

NN_formulate!(MILP_bt_10th_whole, model_10th_whole, sampling_config_10th_norm.ub, sampling_config_10th_norm.lb; bound_tightening="standard", compress=false, silent=false)

@objective(MILP_bt_10th_whole, Max, MILP_bt_10th_whole[:x][5,10])
@constraint(MILP_bt_10th_whole, MILP_bt_10th_whole[:x][5,12] <= (0.34 - mean_10th_y[12])/std_10th_y[12])

set_attribute(MILP_bt_10th_whole, "TimeLimit", 1800)
unset_silent(MILP_bt_10th_whole)
log_filename = "gurobi_log_10th_whole_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_10th_whole, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_10th_whole)

x_star_10th_whole_norm = [value.(MILP_bt_10th_whole[:x][0,i]) for i in 1:length(MILP_bt_10th_whole[:x][0,:])]
x_star_10th_whole = [value.(MILP_bt_10th_whole[:x][0,i]) for i in 1:length(MILP_bt_10th_whole[:x][0,:])] .* std_10th .+ mean_10th

println("Bypass Fraction:", x_star_10th_whole[1])
println("NG Steam Ratio:", x_star_10th_whole[2])
println("H2 Concentration:", objective_value(MILP_bt_10th_whole) * std_10th_y[10] + mean_10th_y[10])
println("N2 Concentration:", value.(MILP_bt_10th_whole[:x][5,12]) * std_10th_y[12] + mean_10th_y[12])

fig = plot_dual_contours(data_reformer_10th_norm, model_10th_whole, x_star_10th_whole_norm, "sol_pool", sol_pool_x_1st_filtered, [1,2], 10)
Makie.save(joinpath(root, "images/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points 10th Iteration_whole.png"), fig)

#------------------------------ 11th iteration --------------------------#
sampling_configs_11th, _ = generate_resample_configs_mc(sampling_config_10th_norm, reshape(x_star_10th_whole_norm, 2, 1), 0.10, 0.3, mean_10th, std_10th)
_, _, selected_indices_11th, resample_data_time_11th = extract_data_from_given_dataset(x_not_selected, y_not_selected, sampling_configs_11th, complement_indices)

x_11th_added = x[:, selected_indices_11th]
y_11th_added = y[:, selected_indices_11th]
x_11th = x[:, vcat(selected_indices_11th, selected_indices_10th, selected_indices_9th, selected_indices_8th, selected_indices_7th, selected_indices_6th, selected_indices_5th, selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st)]
y_11th = y[:, vcat(selected_indices_11th, selected_indices_10th, selected_indices_9th, selected_indices_8th, selected_indices_7th, selected_indices_6th, selected_indices_5th, selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st)]

data_reformer_11th = NN_Data()
train_data_11th, test_data_11th = Flux.splitobs((x_11th, y_11th), at = 0.8)
data_reformer_11th.x_train = Float32.(train_data_11th[1])
data_reformer_11th.y_train = Float32.(train_data_11th[2])
data_reformer_11th.x_test = Float32.(test_data_11th[1])
data_reformer_11th.y_test = Float32.(test_data_11th[2])

@info "option 2: train with the whole combined data"

data_reformer_11th_norm, mean_11th, std_11th, mean_11th_y, std_11th_y = normalise_data(data_reformer_11th, true)
sampling_config_11th_norm = Sampling_Config(
    sampling_config_init.n_samples,
    (sampling_config_init.lb .- vec(mean_11th)) ./ vec(std_11th),
    (sampling_config_init.ub .- vec(mean_11th)) ./ vec(std_11th)
)

config_reformer_11th = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 800, 100, 0)
train_time = @elapsed result_reformer_11th_whole = NN_train(data_reformer_11th_norm, config_reformer_11th, trained_model = model_10th_whole)
NN_results(config_reformer_11th, result_reformer_11th_whole)

model_11th_whole = result_reformer_11th_whole.model
BSON.@save joinpath(@__DIR__, "surrogate_11th_whole.bson") model_11th_whole
BSON.@load joinpath(@__DIR__, "surrogate_11th_whole.bson") model_11th_whole

MILP_bt_11th_whole = Model()
set_optimizer(MILP_bt_11th_whole, Gurobi.Optimizer)
set_silent(MILP_bt_11th_whole)
set_attribute(MILP_bt_11th_whole, "TimeLimit", 10)

NN_formulate!(MILP_bt_11th_whole, model_11th_whole, sampling_config_11th_norm.ub, sampling_config_11th_norm.lb; bound_tightening="standard", compress=false, silent=false)

@objective(MILP_bt_11th_whole, Max, MILP_bt_11th_whole[:x][5,10])
@constraint(MILP_bt_11th_whole, MILP_bt_11th_whole[:x][5,12] <= (0.34 - mean_11th_y[12])/std_11th_y[12])

set_attribute(MILP_bt_11th_whole, "TimeLimit", 1800)
unset_silent(MILP_bt_11th_whole)
log_filename = "gurobi_log_11th_whole_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_11th_whole, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_11th_whole)

x_star_11th_whole_norm = [value.(MILP_bt_11th_whole[:x][0,i]) for i in 1:length(MILP_bt_11th_whole[:x][0,:])]
x_star_11th_whole = [value.(MILP_bt_11th_whole[:x][0,i]) for i in 1:length(MILP_bt_11th_whole[:x][0,:])] .* std_11th .+ mean_11th

println("Bypass Fraction:", x_star_11th_whole[1])
println("NG Steam Ratio:", x_star_11th_whole[2])
println("H2 Concentration:", objective_value(MILP_bt_11th_whole) * std_11th_y[10] + mean_11th_y[10])
println("N2 Concentration:", value.(MILP_bt_11th_whole[:x][5,12]) * std_11th_y[12] + mean_11th_y[12])

fig = plot_dual_contours(data_reformer_11th_norm, model_11th_whole, x_star_11th_whole_norm, "sol_pool", sol_pool_x_1st_filtered, [1,2], 10)
Makie.save(joinpath(root, "images/Comparison of Simulator and Surrogate Model Contours with Optimal Solution Points 11th Iteration_whole.png"), fig)

#------------------------------ 12th iteration --------------------------#
sampling_configs_12th, _ = generate_resample_configs_mc(sampling_config_11th_norm, reshape(x_star_11th_whole_norm, 2, 1), 0.10, 0.3, mean_11th, std_11th)
_, _, selected_indices_12th, resample_data_time_12th = extract_data_from_given_dataset(x_not_selected, y_not_selected, sampling_configs_12th, complement_indices)

x_12th_added = x[:, selected_indices_12th]
y_12th_added = y[:, selected_indices_12th]
x_12th = x[:, vcat(selected_indices_12th, selected_indices_11th, selected_indices_10th, selected_indices_9th, selected_indices_8th, selected_indices_7th, selected_indices_6th, selected_indices_5th, selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st)]
y_12th = y[:, vcat(selected_indices_12th, selected_indices_11th, selected_indices_10th, selected_indices_9th, selected_indices_8th, selected_indices_7th, selected_indices_6th, selected_indices_5th, selected_indices_4th, selected_indices_3rd, selected_indices_2nd, selected_indices_1st)]

data_reformer_12th = NN_Data()
train_data_12th, test_data_12th = Flux.splitobs((x_12th, y_12th), at = 0.8)
data_reformer_12th.x_train = Float32.(train_data_12th[1])
data_reformer_12th.y_train = Float32.(train_data_12th[2])
data_reformer_12th.x_test = Float32.(test_data_12th[1])
data_reformer_12th.y_test = Float32.(test_data_12th[2])

@info "option 2: train with the whole combined data"

data_reformer_12th_norm, mean_12th, std_12th, mean_12th_y, std_12th_y = normalise_data(data_reformer_12th, true)
sampling_config_12th_norm = Sampling_Config(
    sampling_config_init.n_samples,
    (sampling_config_init.lb .- vec(mean_12th)) ./ vec(std_12th),
    (sampling_config_init.ub .- vec(mean_12th)) ./ vec(std_12th)
)

config_reformer_12th = NN_Config([2,10,10,10,10,12], [relu, relu, relu, relu, identity], false, 0, 0.0, Adam(0.001, (0.9, 0.999), 1e-07), 1, 800, 100, 0)
train_time = @elapsed result_reformer_12th_whole = NN_train(data_reformer_12th_norm, config_reformer_12th, trained_model = model_11th_whole)
NN_results(config_reformer_12th, result_reformer_12th_whole)

model_12th_whole = result_reformer_12th_whole.model
BSON.@save joinpath(@__DIR__, "surrogate_12th_whole.bson") model_12th_whole
BSON.@load joinpath(@__DIR__, "surrogate_12th_whole.bson") model_12th_whole

MILP_bt_12th_whole = Model()
set_optimizer(MILP_bt_12th_whole, Gurobi.Optimizer)
set_silent(MILP_bt_12th_whole)
set_attribute(MILP_bt_12th_whole, "TimeLimit", 10)

NN_formulate!(MILP_bt_12th_whole, model_12th_whole, sampling_config_12th_norm.ub, sampling_config_12th_norm.lb; bound_tightening="standard", compress=false, silent=false)

@objective(MILP_bt_12th_whole, Max, MILP_bt_12th_whole[:x][5,10])
@constraint(MILP_bt_12th_whole, MILP_bt_12th_whole[:x][5,12] <= (0.34 - mean_12th_y[12])/std_12th_y[12])

set_attribute(MILP_bt_12th_whole, "TimeLimit", 1800)
unset_silent(MILP_bt_12th_whole)
log_filename = "gurobi_log_12th_whole_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_12th_whole, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_12th_whole)

x_star_12th_whole_norm = [value.(MILP_bt_12th_whole[:x][0,i]) for i in 1:length(MILP_bt_12th_whole[:x][0,:])]
x_star_12th_whole = [value.(MILP_bt_12th_whole[:x][0,i]) for i in 1:length(MILP_bt_12th_whole[:x][0,:])] .* std_12th .+ mean_12th

println("Bypass Fraction:", x_star_12th_whole[1])
println("NG Steam Ratio:", x_star_12th_whole[2])
println("H2 Concentration:", objective_value(MILP_bt_12th_whole) * std_12th_y[10] + mean_12th_y[10])
println("N2 Concentration:", value.(MILP_bt_12th_whole[:x][5,12]) * std_12th_y[12] + mean_12th_y[12])

