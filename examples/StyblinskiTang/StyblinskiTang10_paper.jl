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
- Styblinski-Tang function (2 variables)
"""

# set random seed
Random.seed!(1)

# the function we are going to build surrogate for
styblinski_tang(x::Tuple) = 0.5 * sum([xi^4 - 16*xi^2 + 5*xi for xi in x])
global_min_x = fill(-2.90353, 10)

L_bounds_init = fill(-5.0, 10)
U_bounds_init = fill(5.0, 10)
sampling_config_init = Sampling_Config(1000, L_bounds_init, U_bounds_init)


data_ST = generate_data(styblinski_tang, sampling_config_init, SobolSample(), 0.8)

config1_ST = NN_Config([10,50,50,1], [relu, relu, identity], false, 0.0, 0.1, Adam(0.01, (0.9, 0.999), 1e-07), 1, 800, 1000, 0)
train_time = @elapsed result_ST = NN_train(data_ST, config1_ST)
NN_results(config1_ST, result_ST)

model_init = result_ST.model
BSON.@save "ST10_model_init.bson" model_init
BSON.@load "ST10_model_init.bson" model_init

@info "bound tightening and compression (Gogeta.jl)"
MILP_bt = Model()
set_optimizer(MILP_bt, Gurobi.Optimizer)
set_silent(MILP_bt)
set_attribute(MILP_bt, "TimeLimit", 10)
build_time = @elapsed compressed_model, removed_neurons, bounds_U, bounds_L = NN_formulate!(MILP_bt, model_init, sampling_config_init.ub, sampling_config_init.lb; bound_tightening="fast", compress=true, silent=false);

@objective(MILP_bt, Min, MILP_bt[:x][3,1])

set_attribute(MILP_bt, "TimeLimit", 3600)
unset_silent(MILP_bt)
log_filename = "gurobi_log_init_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt)
write_to_file(MILP_bt, joinpath(@__DIR__, "model_init_bt.mps"))
MILP_bt = read_from_file(joinpath(@__DIR__, "model_init_bt.mps"))

x_star_init = [value.(MILP_bt[:x][0,i]) for i in 1:length(MILP_bt[:x][0,:])]
# x_star_init_norm = [value.(MILP_bt[:x][0,i]) for i in 1:length(MILP_bt[:x][0,:])] .* std_init .+ mean_init

solution_evaluate(MILP_bt, styblinski_tang)

# store multiple solutions in the solution pool
num_solutions_init_bt = MOI.get(MILP_bt, MOI.ResultCount())
sol_pool_x_init_bt, _ = sol_pool(MILP_bt, num_solutions_init_bt)

# visualise the surrogate model 
fig = plot_dual_contours(data_ST, model_init, x_star_init, "sub-optimal solutions", sol_pool_x_init_bt, [1,2], 1, [-2.903534, -2.903534, -2.903534, -2.903534, -2.903534,-2.903534, -2.903534, -2.903534, -2.903534, -2.903534])
Makie.save(joinpath(root, "images/exp1_init_dual_sol_pool.png"), fig)

#------------ apply Monte Carlo Dropout to the surrogate model ------------#
config1_ST_dp = NN_Config([10,500,200,1], [relu, relu, identity], false, 0.0, 0.1, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, 640, 1000, 0)
train_time = @elapsed result_ST = NN_train(data_ST, config1_ST_dp)
NN_results(config1_ST_dp , result_ST)
model_init_dp = result_ST.model

pred, pred_dist, means, stds, x_top_std, mc_time_init = predict_dist(data_ST, model_init_dp, 100, 50)
fig = plot_dual_contours(data_ST, model_init, x_star_init, "x_top_std", [col for col in eachcol(x_top_std)], [2,3], 1, [-2.903534, -2.903534, -2.903534, -2.903534, -2.903534,-2.903534, -2.903534, -2.903534, -2.903534, -2.903534])

#------------------------------ 1st iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_1st, sampling_config_1st = generate_resample_configs_mc(sampling_config_init, [x_top_std x_star_init hcat(sol_pool_x_init_bt...)], 0.20, 0.4, zeros(Float64, 1, 1), ones(Float64, 1, 1))
sampling_config_1st
data_ST_1st_new = generate_and_combine_data(styblinski_tang, sampling_configs_1st, SobolSample(), 0.8)
data_ST_1st = combine_datasets(data_ST, data_ST_1st_new)
# data_ST_1st_filtered = filter_data_within_bounds(data_ST_1st, sampling_config_1st.lb, sampling_config_1st.ub)

x_1st_added = hcat(data_ST_1st_new.x_train, data_ST_1st_new.x_test)

sampling_config_1st = Sampling_Config(
    sampling_config_1st.n_samples,
    sampling_config_1st.lb,
    sampling_config_1st.ub
)

config_ST_1st = NN_Config([10,500,200,1], [relu, relu, identity], false, 0.1, 0.2, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, size(data_ST_1st.x_train)[2], 5000, 1)
train_time_1st = @elapsed result_ST_1st = NN_train(data_ST_1st, config_ST_1st, trained_model = model_init)
NN_results(config_ST_1st, result_ST_1st)

model_1st = result_ST_1st.model
BSON.@save "ST10_model_1st.bson" model_1st
BSON.@load "ST10_model_1st.bson" model_1st

@info "bound tightening and compression (Gogeta.jl)"
MILP_bt_1st = Model()
set_optimizer(MILP_bt_1st, Gurobi.Optimizer)
set_silent(MILP_bt_1st)
set_attribute(MILP_bt_1st, "TimeLimit", 10)

build_time = @elapsed compressed_model_1st, removed_neurons_1st, bounds_U_1st, bounds_L_1st = NN_formulate!(MILP_bt_1st, model_1st, sampling_config_1st.ub, sampling_config_1st.lb; bound_tightening="fast", compress=true, silent=false)
@objective(MILP_bt_1st, Min, MILP_bt_1st[:x][3,1])

set_attribute(MILP_bt_1st, "TimeLimit", 3600)
unset_silent(MILP_bt_1st)
log_filename = "gurobi_log_1st_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
set_optimizer_attribute(MILP_bt_1st, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_1st)
write_to_file(MILP_bt_1st, joinpath(@__DIR__, "model_1st_bt.mps"))
MILP_bt_1st = read_from_file(joinpath(@__DIR__, "model_1st_bt.mps"))

x_star_1st = [value.(MILP_bt_1st[:x][0,i]) for i in 1:length(MILP_bt_1st[:x][0,:])]
# x_star_1st_norm = [value.(MILP_bt_1st[:x][0,i]) for i in 1:length(MILP_bt_1st[:x][0,:])] .* std_1st_filtered .+ mean_1st_filtered

solution_evaluate(MILP_bt_1st, styblinski_tang)

# store multiple solutions in the solution pool
num_solutions_1st_bt = MOI.get(MILP_bt_1st, MOI.ResultCount())
sol_pool_x_1st_bt, _ = sol_pool(MILP_bt_1st, num_solutions_1st_bt)

# visualise the surrogate model
fig = plot_dual_contours(data_ST_1st, model_1st, x_star_1st, "sub-optimal solutions", sol_pool_x_1st_bt, [1,2], 1, global_min_x)


