# using MATLAB
using Statistics
using Surrogates
using Flux
using CSV, DataFrames

include("../../src/NNSurrogate.jl")
include("../../src/NNJuMP.jl")

"""
- Blade max-temperature simulation model (6 variables)
"""


# mat"addpath(pwd(), '../ThermalAnalysisOfJetEngineTurbineBlade')"  # add the path of the MATLAB code

# # arguments: T_air, T_gas, h_air, h_gas_pressureside, h_gas_suctionside, h_gas_tip
# blade_max_temp(x::NTuple{6, Float64}) = mat"computeMaxTemp($(x[1]), $(x[2]), $(x[3]), $(x[4]), $(x[5]), $(x[6]))"

# bounds = [Float64[120, 900, 20, 40, 30, 10], Float64[180, 1200, 40, 60, 50, 30]]

# data_temp = generate_data(blade_max_temp, bounds, 1000, SobolSample(), 0.8)

# get the location of the script file
root = dirname(@__FILE__)

# a robust representation of the filepath to data file
csv_file_path = joinpath(root, "combined_data.csv")

# read the data from the csv file
data = DataFrame(CSV.File(csv_file_path, header=false))

# Extract x (first 6 columns) and y (7th column)
x = convert(Matrix, data[:, 1:6])' 
y = reshape(data[:, 7], size(x, 2), 1)'

# split the data into train set and test set
train_data, test_data = Flux.splitobs((x, y), at = 0.8)

# convert the data to the format of NN_Data
data_temp = NN_Data()
data_temp.x_train = train_data[1]
data_temp.y_train = train_data[2]
data_temp.x_test = test_data[1]
data_temp.y_test = test_data[2]

# data_temp, μ, σ = normalise_data(data_temp) 

config1_temp = NN_Config([6,50,50,1], [relu, relu, identity], false, 0.01, 0.0, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, 800, 5000)
config2_temp = NN_Config([6,512,256,1], [relu, relu, identity], false, 0.01, 0.0, Adam(), 1, 800, 5000)
configs_temp = [config1_temp, config2_temp]

# trian the nerual net
results_temp = NN_compare(data_temp, configs_temp)
for (configs, results) in results_temp
    println("Layers: $(configs.layer), Epochs: $(configs.epochs), Lambda: $(configs.lambda), Dropout rate: $(configs.keep_prob)")    
    println("    Train Error[MSE, RRMSE, MAPE]: $(results.train_err)")
    println("    Test Error [MSE, RRMSE, MAPE]: $(results.test_err)")
end

plot_learning_curve(config1_temp, results_temp[config1_temp].err_hist)

NN_model = results_temp[config1_temp].model

# bounds of the input layer, and the other layers (arbitrary large big-M)
L_bounds = vcat(Float32[120, 900, 20, 40, 30, 10], fill(Float32(-1e6), 101))
U_bounds = vcat(Float32[180, 1200, 40, 60, 50, 30], fill(Float32(1e6), 101))

# convert the trained nerual net to a JuMP model
MILP_model = JuMP_Model(NN_model, L_bounds, U_bounds)

optimize!(MILP_model)

println(objective_value(MILP_model))
println(value.(MILP_model[:x]))