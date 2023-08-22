using MATLAB
using .NNSurrogate
using Surrogates
using Flux

mat"addpath('C:\\Users\\liuy43\\OneDrive - Aalto University\\Code\\ThermalAnalysisOfJetEngineTurbineBlade')"

# arguments: T_air, T_gas, h_air, h_gas_pressureside, h_gas_suctionside, h_gas_tip
blade_max_temp(x::NTuple{6, Float64}) = mat"computeMaxTemp($(x[1]), $(x[2]), $(x[3]), $(x[4]), $(x[5]), $(x[6]))"

bounds = [Float64[120, 900, 20, 40, 30, 10], Float64[180, 1200, 40, 60, 50, 30]]

data_temp = generate_data(blade_max_temp, bounds, 100, SobolSample(), 0.8)

config1_temp = NN_Config([6,512,1], [relu, identity], false, 1, 0.3, Adam(), 1, 80, 1000)

configs_temp = [config1_temp]

# trian the nerual net
results_temp = NN_compare(data_temp, configs_temp)
for (configs, results) in results_temp
    println("Layers: $(configs.layer), Activation: $(configs.af), Epochs: $(configs.epochs), Train Error: $(results.train_err), Test Error: $(results.test_err)")
end

plot_learning_curve(config1_temp, results_temp[config1_temp].err_hist)
