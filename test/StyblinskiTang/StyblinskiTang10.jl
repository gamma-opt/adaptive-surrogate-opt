# using .NNSurrogate   # To load a module from a locally defined module, a dot needs to be added before the module name like using .ModuleName.
using Surrogates
using Flux
using Random
using Statistics
using JuMP
using BSON
using Dates

include("../../src/NNSurrogate.jl")
include("../../src/NNJuMP.jl")
include("../../src/NNGogeta.jl")
include("../../src/NNOptimise.jl")
include("../../src/MCDropout.jl")

"""
- Styblinski-Tang function (10 variables)
"""

# set random seed
Random.seed!(1)

# define the function we are going to build surrogate for
styblinski_tang(x::Tuple) = 0.5 * sum([xi^4 - 16*xi^2 + 5*xi for xi in x])

#--------------------------------- Initial training ---------------------------------#
# sampling
L_bounds_init = fill(-5.0, 10)
U_bounds_init = fill(5.0, 10)
sampling_config_init = Sampling_Config(10000, L_bounds_init, U_bounds_init)

data_ST = generate_data(styblinski_tang, sampling_config_init, SobolSample(), 0.8)

# normalise the data
data_ST_norm, mean_init, std_init, mean_init_y, std_init_y = normalise_data(data_ST, true)

sampling_config_init_norm = Sampling_Config(
    sampling_config_init.n_samples,
    (sampling_config_init.lb .- vec(mean_init)) ./ vec(std_init),
    (sampling_config_init.ub .- vec(mean_init)) ./ vec(std_init)
)

# provide the configurations
config1_ST = NN_Config([10,128,128,1], [relu, relu, identity], false, 0.0, 0.1, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, 8000, 1000, 0)
# config2_ST = NN_Config([10,512, 1], [relu, identity], false, 1, 0.5, Adam(0.001, (0.9, 0.999)), 5, 140, 500, 0)
# config3_ST = NN_Config([10,512,512,512,512,1], [relu, relu, relu, relu, identity], false, 0.1, 0.5, Adam(), 5, 140, 500, 0)
# configs_ST = [config1_ST]
result_ST = NN_train(data_ST_norm, config1_ST)
NN_results(config1_ST, result_ST)
plot_learning_curve(config1_ST, result_ST.err_hist)

# get the trained neural network model from the results
model_init = result_ST.model

# save/load the trained model
BSON.@save joinpath(@__DIR__, "surrogate_init.bson") model_init
BSON.@load joinpath(@__DIR__, "surrogate_init.bson") model_init

#------------ big-M ------------#
L_bounds = vcat(Float32.(sampling_config_init.lb), fill(Float32(-1e6), sum(config1_ST.layer[2:end])))
U_bounds = vcat(Float32.(sampling_config_init.ub), fill(Float32(1e6), sum(config1_ST.layer[2:end])))

MILP_model = JuMP_Model(model_init, L_bounds, U_bounds)
optimize!(MILP_model)

f_hat, f_true, x_star_init, gap = solution_evaluate(MILP_model, styblinski_tang)

#-----------Gogeta----------#

# convert the surrogate model to a MILP model
@info "bound tightening and compression (Gogeta.jl)"
MILP_bt = Model()
set_optimizer(MILP_bt, Gurobi.Optimizer)
set_silent(MILP_bt)
set_attribute(MILP_bt, "TimeLimit", 10)
build_time = @elapsed compressed_model, removed_neurons, bounds_U, bounds_L = NN_formulate!(MILP_bt, model_init, sampling_config_init_norm.ub, sampling_config_init_norm.lb; bound_tightening="fast", compress=true, silent=false);

@objective(MILP_bt, Min, MILP_bt[:x][3,1])

set_attribute(MILP_bt, "TimeLimit", 1800)
unset_silent(MILP_bt)
log_filename = "gurobi_log_init_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt)
write_to_file(MILP_bt, joinpath(@__DIR__, "model_init_bt.mps"))
MILP_bt = read_from_file(joinpath(@__DIR__, "model_init_bt.mps"))

x_star_init_norm = [value.(MILP_bt[:x][0,i]) for i in 1:length(MILP_bt[:x][0,:])]
x_star_init = [value.(MILP_bt[:x][0,i]) for i in 1:length(MILP_bt[:x][0,:])] .* std_init .+ mean_init

# store multiple solutions in the solution pool
num_solutions_init_bt = MOI.get(MILP_bt, MOI.ResultCount())
sol_pool_x_init_bt, _ = sol_pool(MILP_bt, num_solutions_init_bt, mean = mean_init, std = std_init)

println("        MIP solution: ", x_star_init)
println("     Objective value: ", objective_value(MILP_bt) * std_init_y + mean_init_y)
println("True objective value: ", styblinski_tang(Tuple(x_star_init)))

# visualise the surrogate model 
fig = plot_dual_contours(data_ST_norm, model_init, x_star_init_norm, "sol_pool", sol_pool_x_init_bt, [1,2], 1)

#------------ apply Monte Carlo Dropout to the surrogate model ------------#
config1_ST_dp = NN_Config([5,512,256,1], [relu, relu, identity], false, 0.0, 0.1, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, 640, 1000, 0)
train_time = @elapsed result_ST = NN_train(data_ST_norm, config1_ST_dp)
NN_results(config1_ST_dp , result_reformer)
model_init_dp = result_ST.model

pred, pred_dist, means, stds, x_top_std = predict_dist(data_ST_norm, model_init_dp, 100, 10)
fig = plot_dual_contours(data_ST_norm, model_init, x_star_init_norm, "x_top_std", [col for col in eachcol(x_top_std)], [1,2], 1)

#------------------------------ 1st iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_1st, sampling_config_1st = generate_resample_configs_mc(sampling_config_init_norm, [x_top_std x_star_init_norm hcat(sol_pool_x_init_bt...)], 0.10, 0.3, mean_init, std_init)

data_ST_1st_new = generate_and_combine_data(styblinski_tang, sampling_configs_1st, SobolSample(), 0.8)
data_ST_1st = combine_datasets(data_ST, data_ST_1st_new)
data_ST_1st_filtered = filter_data_within_bounds(data_ST_1st, sampling_config_1st.lb, sampling_config_1st.ub)

data_ST_1st_filtered_norm, mean_1st_filtered, std_1st_filtered, mean_1st_filtered_y, std_1st_filtered_y = normalise_data(data_ST_1st_filtered, true)

sampling_config_1st_filtered_norm = Sampling_Config(
    sampling_config_1st.n_samples,
    (sampling_config_1st.lb .- vec(mean_1st_filtered)) ./ vec(std_1st_filtered),
    (sampling_config_1st.ub .- vec(mean_1st_filtered)) ./ vec(std_1st_filtered)
)

config1_ST_1st = NN_Config([5,512,256,1], [relu, relu, identity], false, 0.0, 0.0, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, 1052, 1000, 1)
train_time_1st = @elapsed result_ST_1st = NN_train(data_ST_1st_filtered_norm, config1_ST_1st, trained_model = model_init)
NN_results(config1_ST_1st, result_ST_1st)

model_1st = result_ST_1st.model

# convert the surrogate model to a MILP model
MILP_bt_1st = Model()
set_optimizer(MILP_bt_1st, Gurobi.Optimizer)
set_silent(MILP_bt_1st)
set_attribute(MILP_bt_1st, "TimeLimit", 10)

build_time = @elapsed compressed_model_1st, removed_neurons_1st, bounds_U_1st, bounds_L_1st = NN_formulate!(MILP_bt_1st, model_1st, sampling_config_1st_filtered_norm.ub, sampling_config_1st_filtered_norm.lb; bound_tightening="standard", compress=true, silent=false)

@objective(MILP_bt_1st, Min, MILP_bt_1st[:x][3,1])

set_attribute(MILP_bt_1st, "TimeLimit", 1800)
unset_silent(MILP_bt_1st)
log_filename = "gurobi_log_1st_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_1st, "LogFile", joinpath(@__DIR__, log_filename))
solving_time_1st = @elapsed optimize!(MILP_bt_1st)
write_to_file(MILP_bt_1st, joinpath(@__DIR__, "model_1st_bt.mps"))
MILP_bt_1st = read_from_file(joinpath(@__DIR__, "model_1st_bt.mps"))

x_star_1st_norm = [value.(MILP_bt_1st[:x][0,i]) for i in 1:length(MILP_bt_1st[:x][0,:])]
x_star_1st = [value.(MILP_bt_1st[:x][0,i]) for i in 1:length(MILP_bt_1st[:x][0,:])] .* std_1st_filtered .+ mean_1st_filtered

# store multiple solutions in the solution pool
num_solutions_1st_bt = MOI.get(MILP_bt_1st, MOI.ResultCount())
sol_pool_x_1st_bt, _ = sol_pool(MILP_bt_1st, num_solutions_1st_bt, mean = mean_1st_filtered, std = std_1st_filtered)

println("        MIP solution: ", x_star_1st)
println("     Objective value: ", objective_value(MILP_bt_1st) * std_1st_filtered_y + mean_1st_filtered_y)
println("True objective value: ", styblinski_tang(Tuple(x_star_1st)))

# visualise the surrogate model
fig = plot_dual_contours(data_ST_1st_filtered_norm, model_1st, x_star_1st_norm, "sol_pool", sol_pool_x_1st_bt, [1,2], 1)

