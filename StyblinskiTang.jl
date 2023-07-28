using .NNSurrogate
using Surrogates
using Flux
using Random

"""
- Styblinski-Tang function (10 variables)
"""

# set random seed
Random.seed!(1)

# define the function we are going to build surrogate for
styblinski_tang(x::Tuple) = 0.5 * sum([xi^4 - 16*xi^2 + 5*xi for xi in x])

# sampling
data_ST = generate_data(styblinski_tang, [fill(-2.0, 10), fill(2.0, 10)], 140, 60, SobolSample())

# provide the configurations
config1_ST = NN_Config([10,512,256,1], [relu, relu, identity], 1, 0.5, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1000)
config2_ST = NN_Config([10,512, 1], [relu, identity], 1, 0.5, Adam(0.001, (0.9, 0.999)), 500)
config3_ST = NN_Config([10,512,512,512,512,1], [relu, relu, relu, relu, identity], 0.1, 0.5, Adam(), 500)
configs_ST = [config1_ST, config2_ST, config3_ST]

# trian the nerual net
results_ST = NN_compare(data_ST, configs_ST)
for (configs, results) in results_ST
    println("Layers: $(configs.layer), Activation: $(configs.af), Epochs: $(configs.epochs), Train Error: $(results.train_err), Test Error: $(results.test_err)")
end

plot_learning_curve(config1_ST, results_ST[config1_ST].loss_hist)