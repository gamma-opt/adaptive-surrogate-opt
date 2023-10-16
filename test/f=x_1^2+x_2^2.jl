using .NNSurrogate   # To load a module from a locally defined module, a dot needs to be added before the module name like using .ModuleName.
using Surrogates
using Flux
using Random
using Statistics

include("../src/NNJuMP.jl")

"""
- y=x_1^2+x_2^2 (2 variables)
"""

# set random seed
Random.seed!(1)

# define the function we are going to build surrogate for
func(x::Tuple) = sum(xi^2 for xi in x)

# sampling
data = generate_data(func, [Float32[-1, -1], Float32[1.0, 1.0]], 130, SobolSample(), 0.77)

# define the model
model = Chain(
    Dense(2, 128, relu),
    Dense(128, 1),
)

loss(x, y) = Flux.mse(model(x), y)

ps = Flux.params(model)

learning_rate = Float32(0.1)

opt = Descent(learning_rate)

loss_history = []
epochs = 50

for epoch in 1:epochs
    # train model
    Flux.train!(loss, ps, [(data.x_train, data.y_train)], opt)
    # print report
    train_loss = loss(data.x_train, data.y_train)
    push!(loss_history, train_loss)
    println("Epoch = $epoch : Training Loss = $train_loss")
end

y_pred = model(data.x_test)
y_act = data.y_test
test_error = mean(abs2, model(x) - func(x) for x in data.x_test)
mean(abs2, y_pred - y_act)


# provide the configurations
config = NN_Config([2, 128, 1], [relu, identity], false, 0, 0.0, Descent(0.1), 1, 100, 50)
configs = [config]

# trian the nerual net
results = NN_compare(data, configs)
for (configs, results) in results
    println("Layers: $(configs.layer), Activation: $(configs.af), Epochs: $(configs.epochs), Train Error: $(results.train_err), Test Error: $(results.test_err)")
end

plot_learning_curve(config, results[config].err_hist)

model = results[config].model
y_pred = model(data.x_test)
y_act = data.y_test
mean(abs2, y_pred - y_act)

U_bounds = vcat(Float32[1.0, 1.0], fill(Float32(1e3), 129))
L_bounds = vcat(Float32[-1.0, -1.0], fill(Float32(-1e3), 129))

MILP_model = JuMP_Model(model, L_bounds, U_bounds)

optimize!(MILP_model)

println(objective_value(MILP_model))
println(value.(MILP_model[:x])) 