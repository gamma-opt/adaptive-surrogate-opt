using MLJ
import MLJFlux
using Flux
using RDatasets
using MLJTuning

# Load Boston Housing dataset
boston = dataset("MASS", "Boston")

# Preprocess the data
X = DataFrame(boston[:, Not(:MedV)])  # features
y = boston[:, :MedV]                    # labels

# Splitting off a test set
(X, Xtest), (y, ytest) = partition((X, y), 0.7, multi=true);

builder = MLJFlux.@builder begin    # define custom neural network architectures for regression models
    init=Flux.glorot_uniform(rng)   # weight initialization scheme
    Chain(
        Dense(n_in, 64, relu, init=init),
        Dense(64, 32, relu, init=init),
        Dense(32, n_out, init=init),
    )
end

# instantiating a model
NeuralNetworkRegressor = @load NeuralNetworkRegressor pkg=MLJFlux
model = NeuralNetworkRegressor(
    builder=builder,
    rng=123,
    epochs=20
)

range1 = range(model, :epochs, lower=1, upper=10, scale=:log10)
self_tuning_model = TunedModel(model=model,
                               tuning=Grid(goal=10),
                               resampling=CV(nfolds=6),
                               measure=rms,
                               acceleration=CPUThreads(),
                               range=range1)

# training the model
mach = machine(self_tuning_model, X, y)
fit!(mach)
best_model = fitted_params(mach).best_model
