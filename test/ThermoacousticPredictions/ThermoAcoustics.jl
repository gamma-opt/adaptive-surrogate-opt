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
- Flame model with acoustic tools (5 variables)
"""

mat"addpath(genpath(fullfile(pwd, 'test', 'ThermoacousticPredictions')))"  # add the path of the MATLAB code
ArgRs=0; ArgRn=pi   # Phase of reflection coefficient: ArgRs (inlet), ArgRn (outlet)

# [training_Y(cal_index,1),training_Y(cal_index,2)] = Helmholtz_gain_phase('Secant',G(cal_index),phi(cal_index),R_in(cal_index),ArgRs,R_out(cal_index),ArgRn,alpha(cal_index),config_index,s_init)
compute_growth_rate(x::NTuple{5, Float64}) = mat"Helmholtz_gain_phase('Secant', $(x[1]), $(x[2]), $(x[3]), 0, $(x[4]), pi, $(x[5]), 11, 1i*112*2*pi)"

#--------------------------------- Initial training ---------------------------------#
L_bounds_init = [0.5, 0, 0.7, 0.6, 100]
U_bounds_init = [3, pi, 1, 1, 160]
sampling_config_init = Sampling_Config(1000, L_bounds_init, U_bounds_init)

# get the location of the script file
root = dirname(@__FILE__)

# a robust representation of the filepath to data file
csv_file_path = joinpath(root, "data_init.csv")

# read the data from the csv file
rdata = DataFrame(CSV.File(csv_file_path, header=false))

# Extract x (first 5 columns) and y (6th column)
x = Matrix(rdata[:, 1:5])'
y = reshape(rdata[:, 6], :, 1)'


# split the data into train set and test set
train_data, test_data = Flux.splitobs((x, y), at = 0.8)

# convert the data to the format of NN_Data
data_flame = NN_Data()
data_flame.x_train = Float64.(train_data[1])
data_flame.y_train = Float64.(train_data[2])
data_flame.x_test = Float64.(test_data[1])
data_flame.y_test = Float64.(test_data[2])

config1_flame = NN_Config([5,2048,1024,1024,512,1], [relu, relu, relu, relu, identity], false, 0.1, 0.01, Adam(), 1, 800, 5000, 0)

# trian the nerual net
result_flame = NN_train(data_flame, config1_flame)
NN_results(config1_flame, result_flame)
plot_learning_curve(config1_flame, result_flame.err_hist)

model_init = result_flame.model
BSON.@save "flame_init.bson" model_init
BSON.@load "flame_init.bson" model_init

# bounds of the input layer, and the other layers (arbitrary large big-M)
L_bounds = vcat(Float32.(sampling_config_init.lb), fill(Float32(-1e6), 129), sum(config1_flame.layer[2:end]))
U_bounds = vcat(Float32.(sampling_config_init.ub), fill(Float32(1e6), 129), sum(config1_flame.layer[2:end]))

# convert the trained nerual net to a JuMP model
MILP_model = JuMP_Model(model_init, L_bounds, U_bounds)
optimize!(MILP_model)

f_hat, f_true, x_star_init, gap = solution_evaluate(MILP_model, compute_growth_rate)


