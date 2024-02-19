using MATLAB
using Statistics
using Surrogates
using Flux
using CSV, DataFrames
using BSON
using Plots

include("../../src/NNSurrogate.jl")
include("../../src/NNJuMP.jl")
include("../../src/NNOptimise.jl")
include("../../src/MCDropout.jl")

"""
- Blade max-temperature simulation model (6 variables)
"""

mat"addpath(pwd(), '../ThermalAnalysisOfJetEngineTurbineBlade')"  # add the path of the MATLAB code

# arguments: T_air, T_gas, h_air, h_gas_pressureside, h_gas_suctionside, h_gas_tip
blade_max_temp(x::NTuple{6, Float64}) = mat"computeMaxTemp($(x[1]), $(x[2]), $(x[3]), $(x[4]), $(x[5]), $(x[6]))"

#--------------------------------- Initial training ---------------------------------#

# bounds = [Float64[120, 900, 20, 40, 30, 10], Float64[180, 1200, 40, 60, 50, 30]]
L_bounds_init = [120, 900, 20, 40, 30, 10]
U_bounds_init = [180, 1200, 40, 60, 50, 30]
sampling_config_init = Sampling_Config(1000, L_bounds_init, U_bounds_init)

# data_temp = generate_data(blade_max_temp, bounds, 1000, SobolSample(), 0.8)

# get the location of the script file
root = dirname(@__FILE__)

# a robust representation of the filepath to data file
csv_file_path = joinpath(root, "combined_data.csv")

# read the data from the csv file
data = DataFrame(CSV.File(csv_file_path, header=false))

# Extract x (first 6 columns) and y (7th column)
x = Matrix(data[:, 1:6])' 
y = reshape(data[:, 7], :, 1)'

# split the data into train set and test set
train_data, test_data = Flux.splitobs((x, y), at = 0.8)

# convert the data to the format of NN_Data
data_temp = NN_Data()
data_temp.x_train = Float64.(train_data[1])
data_temp.y_train = Float64.(train_data[2])
data_temp.x_test = Float64.(test_data[1])
data_temp.y_test = Float64.(test_data[2])

config1_temp = NN_Config([6,50,50,1], [relu, relu, identity], false, 0.01, 0.1, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, 800, 5000, 0)

# trian the nerual net
result_temp = NN_train(data_temp, config1_temp)
NN_results(config1_temp, result_temp)

model_init = result_temp.model
BSON.@save "blade_temp_init.bson" model_init
BSON.@load "blade_temp_init.bson" model_init

# bounds of the input layer, and the other layers (arbitrary large big-M)
L_bounds = vcat(Float32.(sampling_config_init.lb), fill(Float32(-1e6), 101), sum(config1_temp.layer[2:end]))
U_bounds = vcat(Float32.(sampling_config_init.ub), fill(Float32(1e6), 101), sum(config1_temp.layer[2:end]))

# convert the trained nerual net to a JuMP model
MILP_model = JuMP_Model(model_init, L_bounds, U_bounds)
optimize!(MILP_model)

f_hat, f_true, x_star_init, gap = solution_evaluate(MILP_model, blade_max_temp)

#------------ optional: apply Monte Carlo Dropout if dropout_rate > 0

pred_dist = predict_dist(data_temp, model_init, 100)
pred_point, sd = predict_point(data_temp, model_init, 100)
pred_rm = remove_outliers(data_temp, pred_dist)

kdeplot(pred_dist[1], pred_point[1])
kdeplot(pred_rm[1], pred_point[1])

# expected prediction error
pred_ee = expected_prediction_error(data_temp, model_init, 100)
kdeplot(pred_ee, mean(pred_ee))
histogram(data_temp.x_test[1, :], weights = pred_ee, bins = 200, xlabel = "x_test[1]", ylabel = "pred_ee", legend=false)
x_belows, x_aboves = find_max_ee(data_temp, model_init, x_star_init, 100)

# expected improvement
pred_ei = expected_improvement(data_temp, model_init, x_star_init, 100)
kdeplot(pred_ei, mean(pred_ei))
histogram(data_temp.x_test[1, :], weights = pred_ei, bins = 200, xlabel = "x_test[1]", ylabel = "pred_ei", legend=false)
x_belows, x_aboves = find_max_ei(data_temp, model_init, x_star_init, 100)


#--------------------------------- 1st iteration ---------------------------------#

#-----strategy 1: fixed percentage of the search space

# generate new samples around x_star
sampling_config_1st = generate_resample_config(sampling_config_init, x_star_init, 0.75, ("fixed_percentage_density", 1.0), "fixed_percentage")

# read the new generated data based on the new sampling configuration
csv_file_path = joinpath(root, "combined_data_1st_fp.csv")
data = DataFrame(CSV.File(csv_file_path, header=false))

# Extract x (first 6 columns) and y (7th column)
x = Matrix(data[:, 1:6])' 
y = reshape(data[:, 7], :, 1)'

# split the data into train set and test set
train_data, test_data = Flux.splitobs((x, y), at = 0.8)

# convert the data to the format of NN_Data
data_temp_1st = NN_Data()
data_temp_1st.x_train = Float32.(train_data[1])
data_temp_1st.y_train = Float32.(train_data[2])
data_temp_1st.x_test = Float32.(test_data[1])
data_temp_1st.y_test = Float32.(test_data[2])

config1_temp = NN_Config([6,50,50,1], [relu, relu, identity], false, 0.01, 0.0, Flux.Optimise.Optimiser(Adam(0.01, (0.9, 0.999)), ExpDecay(0.1)), 1, round(Int, sampling_config_1st.n_samples*0.8), 5000, 1)
# config2_temp = NN_Config([6,512,256,1], [relu, relu, identity], false, 0.01, 0.0, Adam(), 1, 400, 5000, 0)
result_temp = NN_train(data_temp, config1_temp, trained_model = model_init)
NN_results(config1_temp, result_temp)
x_max_mse, y_max_mse = find_max_mse(data_temp, result_temp.model)

model_1st = result_temp.model
BSON.@save "blade_temp_1st.bson" model_1st
BSON.@load "blade_temp_1st.bson" model_1st

L_bounds = vcat(Float32.(sampling_config_1st.lb), fill(Float32(-1e6), sum(config1_temp.layer[2:end])))
U_bounds = vcat(Float32.(sampling_config_1st.ub), fill(Float32(1e6), sum(config1_temp.layer[2:end])))

MILP_model_1st = rebuild_JuMP_Model(model_1st, MILP_model, config1_temp.freeze, L_bounds, U_bounds)
warmstart_JuMP_Model(MILP_model_1st, x_star_init)
optimize!(MILP_model_1st)

f_hat_1st, f_true_1st, x_star_1st, gap_1st = solution_evaluate(MILP_model_1st, blade_max_temp)


#-----strategy 2: error based resampling

x_belows, x_aboves = find_max_errs(data_temp, model_init, x_star_init)
sampling_config_1st_eb = generate_resample_config(sampling_config_init, x_star_init, 1.05, ("fixed_percentage_density", 1.0), "error_based", x_below = x_belows, x_above = x_aboves)


#-----strategy 3: sagmented error based resampling
plots_array = plot_segmented_errs(data_temp, model_init, x_star_init, 50)
# Combine all the individual plots into one composite plot
plot(plots_array..., layout=(10, 1), size=(500, 200 * 10))

x_belows, x_aboves = find_max_segmented_errs(data_temp, model_init, x_star_init, 50)
sampling_config_1st_seb = generate_resample_config(sampling_config_init, x_star_init, 1.05, ("fixed_percentage_density", 1.0), "segmented_error", x_below = x_belows, x_above = x_aboves)