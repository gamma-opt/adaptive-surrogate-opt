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

L_bounds_init = fill(-5.0, 5)
U_bounds_init = fill(5.0, 5)
sampling_config_init = Sampling_Config(1000, L_bounds_init, U_bounds_init)


data_ST = generate_data(styblinski_tang, sampling_config_init, SobolSample(), 0.8)

config1_ST = NN_Config([5,512,256,1], [relu, relu, identity], false, 0.1, 0.1, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, 800, 5000, 0)
train_time = @elapsed result_ST = NN_train(data_ST, config1_ST)
NN_results(config1_ST, result_ST)

model_init = result_ST.model
BSON.@save "ST2_model_init.bson" model_init
BSON.@load "ST2_model_init.bson" model_init

@info "bound tightening and compression (Gogeta.jl)"
MILP_bt = Model()
set_optimizer(MILP_bt, Gurobi.Optimizer)
set_silent(MILP_bt)
set_attribute(MILP_bt, "TimeLimit", 10)
build_time = @elapsed compressed_model, removed_neurons, bounds_U, bounds_L = NN_formulate!(MILP_bt, model_init, sampling_config_init.ub, sampling_config_init.lb; bound_tightening="fast", compress=true, silent=false);

@objective(MILP_bt, Min, MILP_bt[:x][3,1])

set_attribute(MILP_bt, "TimeLimit", 7200)
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
fig = plot_dual_contours(data_ST, model_init, x_star_init, "sub-optimal solutions", sol_pool_x_init_bt, [2,3], 1, [-2.903534, -2.903534, -2.903534, -2.903534, -2.903534])
Makie.save(joinpath(root, "images/exp1_init_dual_sol_pool.png"), fig)

#------------ apply Monte Carlo Dropout to the surrogate model ------------#
config1_ST_dp = NN_Config([5,512,256,1], [relu, relu, identity], false, 0.0, 0.1, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, 640, 1000, 0)
train_time = @elapsed result_ST = NN_train(data_ST, config1_ST_dp)
NN_results(config1_ST_dp , result_ST)
model_init_dp = result_ST.model

pred, pred_dist, means, stds, x_top_std, mc_time_init = predict_dist(data_ST, model_init_dp, 100, 50)
fig = plot_dual_contours(data_ST, model_init, x_star_init, "x_top_std", [col for col in eachcol(x_top_std)], [2,3], 1, [-2.903534, -2.903534, -2.903534, -2.903534, -2.903534])