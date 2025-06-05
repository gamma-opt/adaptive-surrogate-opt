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
using CairoMakie

include("../../src/NNSurrogate.jl")
include("../../src/NNJuMP.jl")
include("../../src/NNGogeta.jl")
include("../../src/NNOptimise.jl")
include("../../src/MCDropout.jl")

"""
- Rastrigin function (10 variables)
"""

root = dirname(@__FILE__)

# set random seed
Random.seed!(1)

# the function we are going to build surrogate for
rastrigin(x::Tuple) = 10*length(x) + sum([xi^2 - 10*cos(2π*xi) for xi in x])
global_min_x = fill(0.0, 5)

L_bounds_init = fill(-5.12, 5)
U_bounds_init = fill(5.12, 5)
sampling_config_init = Sampling_Config(2000, L_bounds_init, U_bounds_init)

start_time = time()

data_ST = generate_data(rastrigin, sampling_config_init, SobolSample(), 0.8)

config1_ST = NN_Config([5,50,50,1], [relu, relu, identity], false, 0.0, 0.0, Adam(0.01, (0.9, 0.999), 1e-07), 1, 800, 500, 0)
train_time = @elapsed result_ST = NN_train(data_ST, config1_ST)
NN_results(config1_ST, result_ST)

model_init = result_ST.model
BSON.@save joinpath(@__DIR__,"R5_model_init.bson") model_init
# BSON.@load "ST10_model_init.bson" model_init

@info "bound tightening and compression (Gogeta.jl)"
MILP_bt = Model()
set_optimizer(MILP_bt, Gurobi.Optimizer)
set_silent(MILP_bt)
set_attribute(MILP_bt, "TimeLimit", 10)
build_time = @elapsed compressed_model, removed_neurons, bounds_U, bounds_L = NN_formulate!(MILP_bt, model_init, sampling_config_init.ub, sampling_config_init.lb; bound_tightening="fast", compress=true, silent=false);

@objective(MILP_bt, Min, MILP_bt[:x][3,1])

set_attribute(MILP_bt, "TimeLimit", 1800)
unset_silent(MILP_bt)
# log_filename = "gurobi_log_init_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
# set_optimizer_attribute(MILP_bt, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt)
# write_to_file(MILP_bt, joinpath(@__DIR__, "model_init_bt.mps"))

x_star_init = [value.(MILP_bt[:x][0,i]) for i in 1:length(MILP_bt[:x][0,:])]
# x_star_init_norm = [value.(MILP_bt[:x][0,i]) for i in 1:length(MILP_bt[:x][0,:])] .* std_init .+ mean_init

solution_evaluate(MILP_bt, rastrigin)

# store multiple solutions in the solution pool
num_solutions_init_bt = MOI.get(MILP_bt, MOI.ResultCount())
sol_pool_x_init_bt, _ = sol_pool(MILP_bt, num_solutions_init_bt)

# visualise the surrogate model 
fig = plot_dual_contours(data_ST, model_init, x_star_init, "sub-optimal solutions", sol_pool_x_init_bt, [1,2], 1, global_min_x)
CairoMakie.activate!()
Makie.save(joinpath(root, "images_pdf/appendix_init_dual_sol_pool.pdf"), fig)

#------------ apply Monte Carlo Dropout to the surrogate model ------------#
config1_ST_dp = NN_Config([5,50,50,1], [relu, relu, identity], false, 0.0, 0.2, Adam(0.1, (0.9, 0.999), 1e-07), 1, 640, 500, 0)
train_time = @elapsed result_ST = NN_train(data_ST, config1_ST_dp)
NN_results(config1_ST_dp , result_ST)
model_init_dp = result_ST.model

pred, pred_dist, means, stds, x_top_std, mc_time_init = predict_dist(data_ST, model_init, 100, 0)
# fig = plot_dual_contours(data_ST, model_init, x_star_init, "points with top 10 highest uncertainty", [col for col in eachcol(x_top_std)], [2,3], 1, global_min_x)
# Makie.save(joinpath(root, "images/appendix_init_dual_uncertainty.png"), fig)

#------------------------------ 1st iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_1st, sampling_config_1st = generate_resample_configs_mc(sampling_config_init, [x_top_std x_star_init hcat(sol_pool_x_init_bt...)], 0.1, 0.3, zeros(Float64, 1, 1), ones(Float64, 1, 1))

data_ST_1st_new = generate_and_combine_data(rastrigin, sampling_configs_1st, SobolSample(), 0.8)
data_ST_1st = combine_datasets(data_ST, data_ST_1st_new)
data_ST_1st_filtered = filter_data_within_bounds(data_ST_1st, sampling_config_1st.lb, sampling_config_1st.ub)

x_1st_added = hcat(data_ST_1st_new.x_train, data_ST_1st_new.x_test)

sampling_config_1st_filtered = Sampling_Config(
    sampling_config_1st.n_samples,
    sampling_config_1st.lb,
    sampling_config_1st.ub
)

config_ST_1st = NN_Config([5,50,50,1], [relu, relu, identity], false, 0.0, 0.05, Adam(0.1, (0.9, 0.999), 1e-07), 1, size(data_ST_1st.x_train)[2], 1000, 0)
train_time_1st = @elapsed result_ST_1st = NN_train(data_ST_1st_filtered, config_ST_1st)
NN_results(config_ST_1st, result_ST_1st)

model_1st = result_ST_1st.model
BSON.@save joinpath(@__DIR__,"R5_model_1st.bson") model_1st
# BSON.@load "ST10_model_1st.bson" model_1st

@info "bound tightening and compression (Gogeta.jl)"
MILP_bt_1st = Model()
set_optimizer(MILP_bt_1st, Gurobi.Optimizer)
set_silent(MILP_bt_1st)
set_attribute(MILP_bt_1st, "TimeLimit", 10)

build_time = @elapsed compressed_model_1st, removed_neurons_1st, bounds_U_1st, bounds_L_1st = NN_formulate!(MILP_bt_1st, model_1st, sampling_config_1st_filtered.ub, sampling_config_1st_filtered.lb; bound_tightening="fast", compress=true, silent=false)
@objective(MILP_bt_1st, Min, MILP_bt_1st[:x][3,1])

set_attribute(MILP_bt_1st, "TimeLimit", 1800)
unset_silent(MILP_bt_1st)
# log_filename = "gurobi_log_1st_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
# set_optimizer_attribute(MILP_bt_1st, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_1st)
# write_to_file(MILP_bt_1st, joinpath(@__DIR__, "model_1st_bt.mps"))
# MILP_bt_1st = read_from_file(joinpath(@__DIR__, "model_1st_bt.mps"))

x_star_1st = [value.(MILP_bt_1st[:x][0,i]) for i in 1:length(MILP_bt_1st[:x][0,:])]
# x_star_1st_norm = [value.(MILP_bt_1st[:x][0,i]) for i in 1:length(MILP_bt_1st[:x][0,:])] .* std_1st_filtered .+ mean_1st_filtered

solution_evaluate(MILP_bt_1st, rastrigin)

# store multiple solutions in the solution pool
num_solutions_1st_bt = MOI.get(MILP_bt_1st, MOI.ResultCount())
sol_pool_x_1st_bt, _ = sol_pool(MILP_bt_1st, num_solutions_1st_bt)

# visualise the surrogate model
fig = plot_dual_contours(data_ST_1st, model_1st, x_star_1st, "sub-optimal solutions", sol_pool_x_1st_bt, [1,2], 1, global_min_x)
Makie.save(joinpath(root, "images_pdf/appendix_1st_dual_sol_pool.pdf"), fig)

#------------ apply Monte Carlo Dropout to the surrogate model ------------#
config_ST_1st_dp = NN_Config([5,50,50,1], [relu, relu, identity], false, 0.0, 0.1, Adam(0.1, (0.9, 0.999), 1e-07), 1, size(data_ST_1st.x_train)[2], 1000, 0)
train_time_1st = @elapsed result_ST_1st = NN_train(data_ST_1st, config_ST_1st_dp)
NN_results(config_ST_1st_dp , result_ST_1st)
model_1st_dp = result_ST_1st.model

pred_1st, pred_dist_1st, means_1st, stds_1st, x_top_std_1st, mc_time_1st = predict_dist(data_ST_1st, model_1st, 100, 2)
# fig = plot_dual_contours(data_ST_1st, model_1st, x_star_1st, "points with top 5 highest uncertainty", [col for col in eachcol(x_top_std_1st)], [2,3], 1, global_min_x)
Makie.save(joinpath(root, "images/appendix_1st_dual_uncertainty.png"), fig)

#------------------------------ 2nd iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_2nd, sampling_config_2nd = generate_resample_configs_mc(sampling_config_1st, [x_top_std_1st x_star_1st hcat(sol_pool_x_1st_bt...)], 0.1, 0.3, zeros(Float64, 1, 1), ones(Float64, 1, 1))

data_ST_2nd_new = generate_and_combine_data(rastrigin, sampling_configs_2nd, SobolSample(), 0.8)
data_ST_2nd = combine_datasets(data_ST_1st, data_ST_2nd_new)
data_ST_2nd_filtered = filter_data_within_bounds(data_ST_2nd, sampling_config_2nd.lb, sampling_config_2nd.ub)

x_2nd_added = hcat(data_ST_2nd_new.x_train, data_ST_2nd_new.x_test)

sampling_config_2nd_filtered = Sampling_Config(
    sampling_config_2nd.n_samples,
    sampling_config_2nd.lb,
    sampling_config_2nd.ub
)

config_ST_2nd = NN_Config([5,50,50,1], [relu, relu, identity], false, 0.0, 0.05, Adam(0.1, (0.9, 0.999), 1e-06), 1, size(data_ST_2nd.x_train)[2], 500, 0)
train_time_2nd = @elapsed result_ST_2nd = NN_train(data_ST_2nd_filtered, config_ST_2nd)
NN_results(config_ST_2nd, result_ST_2nd)

model_2nd = result_ST_2nd.model
BSON.@save joinpath(@__DIR__,"R5_model_2nd.bson") model_2nd

@info "bound tightening and compression (Gogeta.jl)"
MILP_bt_2nd = Model()
set_optimizer(MILP_bt_2nd, Gurobi.Optimizer)
set_silent(MILP_bt_2nd)
set_attribute(MILP_bt_2nd, "TimeLimit", 10)

build_time = @elapsed compressed_model_2nd, removed_neurons_2nd, bounds_U_2nd, bounds_L_2nd = NN_formulate!(MILP_bt_2nd, model_2nd, sampling_config_2nd_filtered.ub, sampling_config_2nd_filtered.lb; bound_tightening="fast", compress=true, silent=false)
@objective(MILP_bt_2nd, Min, MILP_bt_2nd[:x][3,1])

set_attribute(MILP_bt_2nd, "TimeLimit", 1800)
unset_silent(MILP_bt_2nd)
# log_filename = "gurobi_log_2nd_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
# set_optimizer_attribute(MILP_bt_2nd, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_2nd)
# write_to_file(MILP_bt_2nd, joinpath(@__DIR__, "model_2nd_bt.mps"))
# MILP_bt_2nd = read_from_file(joinpath(@__DIR__, "model_2nd_bt.mps"))

x_star_2nd = [value.(MILP_bt_2nd[:x][0,i]) for i in 1:length(MILP_bt_2nd[:x][0,:])]
# x_star_2nd_norm = [value.(MILP_bt_2nd[:x][0,i]) for i in 1:length(MILP_bt_2nd[:x][0,:])] .* std_2nd_filtered .+ mean_2nd_filtered

solution_evaluate(MILP_bt_2nd, rastrigin)

# store multiple solutions in the solution pool
num_solutions_2nd_bt = MOI.get(MILP_bt_2nd, MOI.ResultCount())
sol_pool_x_2nd_bt, _ = sol_pool(MILP_bt_2nd, num_solutions_2nd_bt)

# visualise the surrogate model
fig = plot_dual_contours(data_ST_2nd, model_2nd, x_star_2nd, "sub-optimal solutions", sol_pool_x_2nd_bt, [1,2], 1, global_min_x)
Makie.save(joinpath(root, "images_pdf/appendix_2nd_dual_sol_pool.pdf"), fig)

#------------ apply Monte Carlo Dropout to the surrogate model ------------#
config_ST_2nd_dp = NN_Config([5,50,50,1], [relu, relu, identity], false, 0.0, 0.1, Adam(0.1, (0.9, 0.999), 1e-06), 1, size(data_ST_2nd.x_train)[2], 1000, 0)
train_time_2nd = @elapsed result_ST_2nd = NN_train(data_ST_2nd, config_ST_2nd_dp)
NN_results(config_ST_2nd_dp , result_ST_2nd)
model_2nd_dp = result_ST_2nd.model

pred_2nd, pred_dist_2nd, means_2nd, stds_2nd, x_top_std_2nd, mc_time_2nd = predict_dist(data_ST_2nd, model_2nd, 100, 2)
# fig = plot_dual_contours(data_ST_2nd, model_2nd, x_star_2nd, "points with top 5 highest uncertainty", [col for col in eachcol(x_top_std_2nd)], [2,3], 1, global_min_x)
# Makie.save(joinpath(root, "images/appendix_2nd_dual_uncertainty.png"), fig)

#------------------------------ 3rd iteration --------------------------#
# Resample densely around the points with the highest uncertainty
sampling_configs_3rd, sampling_config_3rd = generate_resample_configs_mc(sampling_config_2nd, [x_top_std_2nd x_star_2nd hcat(sol_pool_x_2nd_bt...)], 0.1, 0.3, zeros(Float64, 1, 1), ones(Float64, 1, 1))

data_ST_3rd_new = generate_and_combine_data(rastrigin, sampling_configs_3rd, SobolSample(), 0.8)
data_ST_3rd = combine_datasets(data_ST_2nd, data_ST_3rd_new)
data_ST_3rd_filtered = filter_data_within_bounds(data_ST_3rd, sampling_config_3rd.lb, sampling_config_3rd.ub)

x_3rd_added = hcat(data_ST_3rd_new.x_train, data_ST_3rd_new.x_test)

sampling_config_3rd_filtered = Sampling_Config(
    sampling_config_3rd.n_samples,
    sampling_config_3rd.lb,
    sampling_config_3rd.ub
)

config_ST_3rd = NN_Config([5,50,50,1], [relu, relu, identity], false, 0.0, 0.05, Adam(0.1, (0.9, 0.999), 1e-06), 1, size(data_ST_3rd.x_train)[2], 1000, 0)
train_time_3rd = @elapsed result_ST_3rd = NN_train(data_ST_3rd, config_ST_3rd)
NN_results(config_ST_3rd, result_ST_3rd)

model_3rd = result_ST_3rd.model

# BSON.@save "ST10_model_3rd.bson" model_3rd
# BSON.@load "ST10_model_3rd.bson" model_3rd

@info "bound tightening and compression (Gogeta.jl)"
MILP_bt_3rd = Model()
set_optimizer(MILP_bt_3rd, Gurobi.Optimizer)
set_silent(MILP_bt_3rd)

build_time = @elapsed compressed_model_3rd, removed_neurons_3rd, bounds_U_3rd, bounds_L_3rd = NN_formulate!(MILP_bt_3rd, model_3rd, sampling_config_3rd.ub, sampling_config_3rd.lb; bound_tightening="fast", compress=true, silent=false)
@objective(MILP_bt_3rd, Min, MILP_bt_3rd[:x][3,1])

set_attribute(MILP_bt_3rd, "TimeLimit", 1800)
unset_silent(MILP_bt_3rd)
# log_filename = "gurobi_log_3rd_bt_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log"
# set_optimizer_attribute(MILP_bt_3rd, "LogFile", joinpath(@__DIR__, log_filename))
solving_time = @elapsed optimize!(MILP_bt_3rd)
# write_to_file(MILP_bt_3rd, joinpath(@__DIR__, "model_3rd_bt.mps"))
# MILP_bt_3rd = read_from_file(joinpath(@__DIR__, "model_3rd_bt.mps"))

x_star_3rd = [value.(MILP_bt_3rd[:x][0,i]) for i in 1:length(MILP_bt_3rd[:x][0,:])]
# x_star_3rd_norm = [value.(MILP_bt_3rd[:x][0,i]) for i in 1:length(MILP_bt_3rd[:x][0,:])] .* std_3rd_filtered .+ mean_3rd_filtered

solution_evaluate(MILP_bt_3rd, rastrigin)

time_elapsed = time() - start_time

# store multiple solutions in the solution pool
num_solutions_3rd_bt = MOI.get(MILP_bt_3rd, MOI.ResultCount())
sol_pool_x_3rd_bt, _ = sol_pool(MILP_bt_3rd, num_solutions_3rd_bt)

# visualise the surrogate model
fig = plot_dual_contours(data_ST_3rd, model_3rd, x_star_3rd, "sub-optimal solutions", sol_pool_x_3rd_bt, [1,2], 1, global_min_x)
Makie.save(joinpath(root, "images_pdf/appendix_3rd_dual_sol_pool.pdf"), fig)

