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

L_bounds_init = fill(-5.0, 2)
U_bounds_init = fill(5.0, 2)
sampling_config_init = Sampling_Config(1000, L_bounds_init, U_bounds_init)


data_ST = generate_data(styblinski_tang, sampling_config_init, SobolSample(), 0.8)

config1_ST = NN_Config([2,512,256,1], [relu, relu, identity], false, 0.1, 0.1, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, 800, 5000, 0)
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

set_attribute(MILP_bt, "TimeLimit", 1800)
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
fig = plot_dual_contours(data_temp, model_init, x_star_init, "sol_pool", sol_pool_x_init_bt, [1,2], 1)
Makie.save(joinpath(root, "images/exp1_init_dual_sol_pool.png"), fig)

MILP_model = JuMP_Model(model_init, L_bounds, U_bounds)
optimize!(MILP_model)
f_hat_init = objective_value(MILP_model)
x_star_init = get_x_star(MILP_model)
f_init = styblinski_tang(Tuple(x_star_init))

# Theoretical optimal solution
x_opt = [-2.90354,-2.90354]
f_opt = styblinski_tang(Tuple(x_opt))

# visualise the surrogate model
x, y = Float32.(-4.0:0.002:-2.0), Float32.(-4.0:0.002:-2.0)
z_true = [styblinski_tang((x1,x2)) for x1 in x, x2 in y]
z_surrogate = [model_init([x1, x2])[1] for x1 in x, x2 in y]

p1 = surface(x, y, z_true, clims = (-80, -20), colorbar = false)
scatter!(p1, [x_opt[1]], [x_opt[2]], [f_opt], label = "")
p2 = surface(x, y, z_surrogate, clims = (-80, -20), colorbar = false)
scatter!(p2, [x_star_init[1]], [x_star_init[2]], [f_hat_init], label = "")
plot(p1, p2, 
    title=["True function" "Surrogate model"], 
    zlims = (-80, -20),
    legend = false
)
savefig("ST2_surrogate_surface.png")

p1 = contour(x, y, z_true, clims = (-80, -20))
scatter!(p1, [x_opt[1]], [x_opt[2]], label = "")
p2 = contour(x, y, z_surrogate, clims = (-80, -20))
scatter!(p2, [x_star_init[1]], [x_star_init[2]], label = "")
plot(p1, p2, 
    title=["True function" "Surrogate model"], 
    legend = false
)
savefig("ST2_surrogate_contour.png")