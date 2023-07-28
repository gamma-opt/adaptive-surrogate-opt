using .NNSurrogate
using Surrogates
using Flux
using Random
"""
- Goldstein-Price function (2 variables)
"""

# set random seed
Random.seed!(1)

# define the function we are going to build surrogate for
goldstein_price(x::Tuple) = (1+(x[1]+x[2]+1)^2*(19-14*x[1]+3*x[1]^2-14*x[2]+6*x[1]*x[2]+3*x[2]^2)) * 
    (30+(2*x[1]-3*x[2])^2*(18-32*x[1]+12*x[1]^2+48*x[2]-36*x[1]*x[2]+27*x[2]^2))

# sampling
data_GP = generate_data(goldstein_price, [Float32[-0.5, -1.5], Float32[0.5, -0.5]], 140, 60, SobolSample())

# provide the configurations
config1_GP = NN_Config([2,1024,1], [relu, identity], 0.01, 0.4, Flux.Optimise.Optimiser(Adam(1, (0.89, 0.899)), ExpDecay(1, 0.08, 1000, 1e-4)), 5000)
config2_GP = NN_Config([2,512,1], [relu, identity], 0.01, 0.4, Adam(), 1000)
config3_GP = NN_Config([2,512,1], [relu, identity], 0.001, 0.4, Adam(), 1000)
configs_GP = [config1_GP, config2_GP, config3_GP]

# trian the nerual net
results_GP = NN_compare(data_GP, configs_GP)
for (configs, results) in results_GP
    println("Layers: $(configs.layer), Activation: $(configs.af), Epochs: $(configs.epochs), Train Error: $(results.train_err), Test Error: $(results.test_err)")
end

plot_learning_curve(config1_GP, results_GP[config1_GP].loss_hist)
