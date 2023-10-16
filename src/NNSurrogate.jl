# module NNSurrogate

using Plots
using Statistics
using Surrogates
using Flux
using Flux: params, train!
using LinearAlgebra
# using SurrogatesFlux
# using Lathe.preprocess: TrainTestSplit

# export NN_Data, NN_Config, NN_Result
# export generate_data, normalise_data, load_data
# export NN_train, NN_compare, plot_learning_curve

mutable struct NN_Data
    x_train::Matrix
    y_train::Matrix
    x_test::Matrix
    y_test::Matrix
    NN_Data() = new()
end

mutable struct NN_Config
    layer::Vector{Int}   # layers' sizes
    af::Vector           # activation functions
    batch_norm::Bool     # whether to use batch normalization
    lambda::Float64      # regularization parameter (need to be tuned)
    keep_prob::Float64   # dropout rate
    opt::Any             # optimiser
    k::Int               # Number of splits
    batch_size::Int      # batch size
    epochs::Int          # number of epochs
end

mutable struct NN_Result
    model::Chain        # the trained model
    err_hist::Vector    # loss history of the training
    train_err::Vector   # training error
    test_err::Vector    # testing error
    NN_Result() = new()
end

# define the function converting Vector{Tuple} to Matrix
T2M(x::Vector) = reshape(collect(Iterators.flatten(x)), length(x[1]), length(x))

# define the function to generate data for the training and testing
function generate_data(f::Function, bounds::Vector, n_samples::Int, sampling_method::Any, splits::Float64)
    
    # initialise the data
    data = NN_Data()

    # sample and generate the train set 
    x_tuple = sample(n_samples, bounds..., sampling_method)
    x = T2M(x_tuple)  # convert Vector{Tuple} to Matrix
    y = T2M(f.(x_tuple))  # generate the corresponding output

    # split the data into train set and test set
    cv_data, test_data = Flux.splitobs((x, y), at = splits)

    # assign the data to the struct
    data.x_train = cv_data[1]
    data.y_train = cv_data[2]
    data.x_test = test_data[1]
    data.y_test = test_data[2]

    return data
end

# normalise the data
function normalise_data(data::NN_Data) 
    
    μ, σ = mean(data.x_train, dims=2), std(data.x_train, dims=2)  # compute the mean and standard deviation of the training set
    data.x_train = Flux.normalise(data.x_train)  # normalise the training set
    data.x_test = (data.x_test .- μ) ./ σ  # normalise the test set using the mean and standard deviation of the training set
    
    return data
end

# define the loss function with L2 regularization
function loss_l2(x, y, model, lambda)
    
    mse_loss = Flux.mse(model(x), y) # mean squared error
    l2_loss = sum(p -> sum(abs2, p), params(model)) # L2 regularization term
     
    return mse_loss + 0.5 * lambda / length(y) * l2_loss
end

# define the loss function with L1 regularization
function loss_l1(x, y, model, lambda)  

    mse_loss = Flux.mse(model(x), y) # mean squared error
    l1_loss = sum(p -> sum(abs, p), params(model)) # L1 regularization term

    return mse_loss + lambda / length(y) * l1_loss
end

# define the relative root mean square error (RRMSE)
function loss_rrmse(x, y, model)

    mse_loss = Flux.mse(model(x), y) # mean squared error
    mse_div = sum(model(x).^2) # divsor of the relative error 

    return sqrt(mse_loss / mse_div)
end

# define the mean absolute percentage error (MAPE)
function loss_mape(x, y, model)

    mape_loss = mean(abs.((model(x) .- y) ./ y)) 
    return mape_loss

end

# define the function processing the training of Neural Network
function NN_train(data::NN_Data, c::NN_Config)
    
    # initialise the reultes
    result = NN_Result()
    train_err_batch = Float64[]
    train_err_epoch = Float64[]
    train_err_fold = Float64[]
    
    # normalise the data
    data = normalise_data(data) 

    # define model architecture
    chain = []
    push!(chain, Dense(c.layer[1], c.layer[2]))
    if c.batch_norm
        push!(chain, BatchNorm(c.layer[2], c.af[1]))    # add batch normalization layer
    else
        push!(chain, c.af[1])   # add activation function
    end
    push!(chain, Dropout(c.keep_prob) )   # add dropout layer to prevent overfitting   
    for i in 2:length(c.af)
        push!(chain, Dense(c.layer[i], c.layer[i+1]))  # create a traditional fully connected layer
        if c.batch_norm
            push!(chain, BatchNorm(c.layer[i+1], c.af[i]))   # add batch normalization layer to stabilize the network
        else
            push!(chain, c.af[i])   # add activation function
        end 
    end
    result.model = Chain(chain...)

    # track parameters
    ps = params(result.model)

    # define the loss function with L2 regularization
    loss(x,y) = loss_l2(x, y, result.model, c.lambda)
    
    # define the training process
    if c.k > 1  # k must to be within 2:160
        # imply k-fold cross validation
        for (train_data, val_data) in Flux.kfolds((data.x_train, data.y_train), c.k)
            for epoch in 1:c.epochs            
                # mini-batch iterations 
                for (x, y) in Flux.eachobs(train_data, batchsize = c.batch_size)
                    train!(loss, ps, [(x, y)], c.opt)
                    batch_err = loss(x, y)
                    push!(train_err_batch, batch_err)
                end
                push!(train_err_epoch, train_err_batch[end])                                 
            end
            push!(train_err_fold, train_err_epoch[end])           
        end      
        result.err_hist = vec(mean(reshape(train_err_epoch, c.k, c.epochs), dims = 1))
        result.train_err = [mean(train_err_fold), loss_rrmse(data.x_train, data.y_train, result.model), loss_mape(data.x_train, data.y_train, result.model)]
    else   
        # imply no cross validation
        for epoch in 1:c.epochs
            # mini-batch iterations
            for (x, y) in Flux.eachobs((data.x_train, data.y_train), batchsize = c.batch_size)
                train!(loss, ps, [(x, y)], c.opt)
                batch_err = loss(x, y)
                push!(train_err_batch, batch_err)
            end
            push!(train_err_epoch, train_err_batch[end])                                  
        end
        trainmode!(result.model)
        result.err_hist = train_err_epoch
        result.train_err = [train_err_epoch[end], loss_rrmse(data.x_train, data.y_train, result.model), loss_mape(data.x_train, data.y_train, result.model) ]
    end   

    return result
end

# define a function to compare train errors and test errors within different NN configs
function NN_compare(data::NN_Data, configs::Vector{NN_Config})
    
    x_test = data.x_test
    y_test = data.y_test

    n = length(configs)
    results = fill(NN_Result(), n)

    # generate a dict to store the configs and corresponding errors
    results_cp = Dict()

    for i in 1:n
        results[i] = NN_train(data, configs[i])
        testmode!(results[i].model)  # switch to test mode
        results[i].test_err = [Flux.mse(results[i].model(x_test), y_test), loss_rrmse(x_test, y_test, results[i].model), loss_mape(x_test, y_test, results[i].model)]
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

# end # module