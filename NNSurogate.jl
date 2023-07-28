module NNSurrogate

using Plots
using Statistics
using Surrogates
using Flux
using Flux: params, train!
using LinearAlgebra
# using SurrogatesFlux
# using Lathe.preprocess: TrainTestSplit
# using DataFrames

export NN_Data, NN_Config, NN_Result
export generate_data, NN_train, NN_compare, plot_learning_curve

mutable struct NN_Data
    bounds::Vector
    x_train::Matrix
    y_train::Matrix
    x_test::Matrix
    y::Matrix
    NN_Data() = new()
end

mutable struct NN_Config
    layer::Vector{Int}   # layers' sizes
    af::Vector           # activation functions
    lambda::Float64      # regularization parameter (need to be tuned)
    keep_prob::Float64   # dropout rate
    opt::Any             # optimiser
    epochs::Int          # number of epochs
end

mutable struct NN_Result
    loss_hist::Vector
    model::Chain
    train_err::Float64
    test_err::Float64
    NN_Result() = new()
end

# define the function converting Vector{Tuple} to Matrix
T2M(x::Vector) = reshape(collect(Iterators.flatten(x)), length(x[1]), length(x))

# define the function to generate data for the training and testing
function generate_data(f::Function, bounds::Vector, n_samples_train::Int, n_samples_test::Int, sampling_method::Any)
    
    # initialise the data
    data = NN_Data()

    # sample and generate the train set 
    x_train_tuple = sample(n_samples_train, bounds..., sampling_method)
    data.x_train = Flux.normalise(T2M(x_train_tuple))
    data.y_train = T2M(f.(x_train_tuple))  

    # provide the test set
    x_test_tuple = sample(n_samples_test, bounds..., sampling_method)
    data.y = T2M(f.(x_test_tuple))

    # normalize the test set using the statistics computed from the training set
    mean_train = mean(T2M(x_train_tuple), dims=2)  # mean
    std_train = std(T2M(x_train_tuple), dims=2)    # standard deviation
    data.x_test = (T2M(x_test_tuple) .- mean_train) ./ std_train 

    data.bounds = bounds

    return data
end

# define the loss function with L2 regularization
function loss_l2(x, y, model, lambda)
    # root mean squared error

    mse_loss = Flux.huber_loss(model(x), y) # huber loss
    
    l2_loss = sum(p -> sum(abs2, p), params(model)) # L2 regularization term
     
    return mse_loss + 0.5 * lambda / length(y) * l2_loss
end

# define the function processing the training of Neural Network
function NN_train(data::NN_Data, c::NN_Config)
    
    # initialise the reultes
    result = NN_Result()
    result.loss_hist = []
    result.train_err = 0.0

    # define model architecture
    chain = []
    push!(chain, Dense(c.layer[1], c.layer[2], c.af[1]), Dropout(c.keep_prob))  # add dropout layer to avoid overfitting
    for i in 2:length(c.af)
        push!(chain, Dense(c.layer[i], c.layer[i+1], c.af[i]))
    end
    result.model = Chain(chain...)

    # track parameters
    ps = params(result.model)

    # define the loss function with L2 regularization
    loss(x,y) = loss_l2(x, y, result.model, c.lambda)
    
    # train model
    for epoch in 1:c.epochs
        train!(loss, ps, [(data.x_train, data.y_train)], c.opt)
        train_loss = loss(data.x_train, data.y_train)
        # println("Epoch = $epoch : Training Loss = $train_loss")
        push!(result.loss_hist, train_loss)
    end    
    result.train_err = result.loss_hist[end]

    return result
end

# define a function to compare train errors and test errors within different NN configs
function NN_compare(data::NN_Data, configs::Vector{NN_Config})
    
    x_test = data.x_test
    y = data.y

    n = length(configs)
    results = fill(NN_Result(), n)

    # generate a dict to store the configs and corresponding errors
    results_cp = Dict()

    for i in 1:n
        results[i] = NN_train(data, configs[i])
        results[i].test_err = Flux.huber_loss(results[i].model(x_test), y)
        results_cp[(configs[i])] = results[i]
    end
    
    return results_cp
end

# define a function to plot the learning curve based on the loss history
function plot_learning_curve(config::NN_Config, loss_hist::Vector)
    
    # initialize plot
    gr(size = (600, 600))

    # plot learning curve
    plot(1:config.epochs, loss_hist,
        xlabel = "Epochs",
        ylabel = "Loss",
        title = "Learning Curve",
        label = "Layers: $(config.layer), Activation: $(config.af)",
        color = :blue,
        linewidth = 2
    )
end

end # module

