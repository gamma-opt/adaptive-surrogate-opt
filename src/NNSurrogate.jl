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

mutable struct Sampling_Config
    n_samples::Int
    lb::Vector
    ub::Vector
end

mutable struct NN_Config
    layer::Vector{Int}   # layers' sizes
    af::Vector           # activation functions
    batch_norm::Bool     # whether to use batch normalization
    lambda::Float64      # regularization parameter (need to be tuned)
    dropout_rate::Float64   # dropout rate
    opt::Any             # optimiser
    k::Int               # Number of splits
    batch_size::Int      # batch size
    epochs::Int          # number of epochs
    freeze::Int          # number of initial layers to freeze
end

mutable struct NN_Result
    model::Chain        # the trained model
    err_hist::Vector    # loss history of the training
    train_err::Vector   # training error
    test_err::Vector    # testing error
    NN_Result() = new()
end

# convert Vector{Tuple} to Matrix
T2M(x::Vector) = reshape(collect(Iterators.flatten(x)), length(x[1]), length(x))

# generate data for the training and testing
function generate_data(f::Function, sampling_config::Sampling_Config, sampling_method::Any, splits::Float64)
    
    # initialise the data
    data = NN_Data()

    # sample and generate the train set 
    x_tuple = Surrogates.sample(sampling_config.n_samples, [sampling_config.lb, sampling_config.ub]..., sampling_method)
    x = T2M(x_tuple)  # convert Vector{Tuple} to Matrix
    y = T2M(f.(x_tuple))  # generate the corresponding output

    # split the data into train set and test set
    cv_data, test_data = Flux.splitobs((x, y), at = splits)

    # assign the data to the struct
    data.x_train = Float32.(cv_data[1])
    data.y_train = Float32.(cv_data[2])
    data.x_test = Float32.(test_data[1])
    data.y_test = Float32.(test_data[2])

    return data
end

# normalise the data
function normalise_data(data::NN_Data) 
    
    μ, σ = mean(data.x_train, dims=2), std(data.x_train, dims=2)  # compute the mean and standard deviation of the training set
    data.x_train = Flux.normalise(data.x_train)  # normalise the training set
    data.x_test = (data.x_test .- μ) ./ σ  # normalise the test set using the mean and standard deviation of the training set
    
    return data, μ, σ
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

# process the training of Neural Network
function NN_train(data::NN_Data, c::NN_Config; trained_model::Chain = Chain())
    
    # initialise the reultes
    result = NN_Result()
    train_err_batch = Float32[]
    train_err_epoch = Float32[]
    train_err_fold = Float32[]

    # define model architecture
    chain = []
    push!(chain, Dense(c.layer[1], c.layer[2]))
    if c.batch_norm
        push!(chain, BatchNorm(c.layer[2], c.af[1]))    # add batch normalization layer
    else
        push!(chain, c.af[1])   # add activation function
    end
    push!(chain, Dropout(c.dropout_rate) )   # add dropout layer to prevent overfitting   
    for i in 2:length(c.af)
        push!(chain, Dense(c.layer[i], c.layer[i+1]))  # create a traditional fully connected layer
        if c.batch_norm
            push!(chain, BatchNorm(c.layer[i+1], c.af[i]))   # add batch normalization layer to stabilize the network
        else
            push!(chain, c.af[i])   # add activation function
        end 
    end

    # track parameters
    if c.freeze > 0     
        result.model = trained_model                    # use the trained model as the initial model
        ps = params(result.model[(c.freeze+1)*2:end])   # freeze the first few layers
    else
        result.model = Chain(chain...)                  # create a new model    
        ps = params(result.model)                       
    end

    # set the loss function
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
        
        # switch to test mode
        testmode!(result.model)  
        result.test_err = [Flux.mse(result.model(data.x_test), data.y_test), loss_rrmse(data.x_test, data.y_test, result.model), loss_mape(data.x_test, data.y_test, result.model)]
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
        
        result.err_hist = train_err_epoch
        result.train_err = [train_err_epoch[end], loss_rrmse(data.x_train, data.y_train, result.model), mean(abs.((result.model(data.x_train) .- data.y_train) ./ data.y_train)) ]

        # switch to test mode
        testmode!(result.model)
        result.test_err = [Flux.mse(result.model(data.x_test), data.y_test), loss_rrmse(data.x_test, data.y_test, result.model), loss_mape(data.x_test, data.y_test, result.model)]
    end   

    return result
end

# print the results of the training
function NN_results(c::NN_Config, result::NN_Result)   
    println("Layers: $(c.layer), Epochs: $(c.epochs), Lambda: $(c.lambda), Dropout rate: $(c.dropout_rate)")    
    println("    Train Error[MSE, RRMSE, MAPE]: $(result.train_err)")
    println("    Test Error [MSE, RRMSE, MAPE]: $(result.test_err)")
end


# compare train errors and test errors within different NN configs
function NN_compare(data::NN_Data, configs::Vector{NN_Config}; trained_model::Chain = Chain())
    
    x_test = data.x_test
    y_test = data.y_test

    n = length(configs)
    results = fill(NN_Result(), n)

    # generate a dict to store the configs and corresponding errors
    results_cp = Dict()

    for i in 1:n
        results[i] = NN_train(data, configs[i], trained_model = trained_model)
        testmode!(results[i].model)  # switch to test mode
        results[i].test_err = [Flux.mse(results[i].model(x_test), y_test), loss_rrmse(x_test, y_test, results[i].model), loss_mape(x_test, y_test, results[i].model)]
        results_cp[(configs[i])] = results[i]
    end
    
    return results_cp
end


function find_max_errs(data::NN_Data, model::Chain, x_star::Vector{Float64})
    
    # Concatenate training and testing data
    x = hcat(data.x_train, data.x_test)
    y = hcat(data.y_train, data.y_test)

    # Calculate the errors (absolute difference)
    errors = abs.(model(x) .- y)

    # Initialize vectors to store the point with maximum error below and above x_star for each dimension
    x_belows = copy(x_star)
    x_aboves = copy(x_star)

    # Iterate through each dimension to find the point with max error below and above x_star
    for i in 1:length(x_star)
        # Identify indices where x is below and above x_star[i]
        below_indices = findall(x[i, :] .< x_star[i])
        above_indices = findall(x[i, :] .> x_star[i])

        # Calculate errors for below and above points
        if !isempty(below_indices)
            below_errors = errors[below_indices]
            max_below_idx = below_indices[argmax(below_errors)]
            x_belows[i] = x[i, max_below_idx]  # Store the i-th dimension of the point with the maximum below error
        else
            x_belows[i] = x_star[i]
        end

        if !isempty(above_indices)
            above_errors = errors[above_indices]
            max_above_idx = above_indices[argmax(above_errors)]
            x_aboves[i] = x[i, max_above_idx]  # Store the i-th dimension of the point with the maximum above error
        else
            x_aboves[i] = x_star[i]
        end
    end

    return x_belows, x_aboves
end


# calculate the segmented errors for a certain dimension
function calculate_segmented_errs(x, errors, x_star, n_segments)
    
    min_val, max_val = minimum(x), maximum(x)
    bin_edges = range(min_val, stop=max_val, length=n_segments+1)

    # Initialize tracking variables for maximum error and corresponding midpoints
    max_error_above = -Inf
    max_error_below = -Inf
    mid_point_above = x_star
    mid_point_below = x_star
    
    # Initialize an array to store errors for all segments
    all_segment_errors = Vector{Vector{Float32}}(undef, n_segments)

    for i = 1:n_segments
        in_segment = (x .>= bin_edges[i]) .& (x .< bin_edges[i+1])
        segment_errors = errors[in_segment]

        # Store segment errors
        all_segment_errors[i] = segment_errors

        # Skip the loop iteration if the segment is empty
        if isempty(segment_errors)
            continue
        end

        segment_mean_error = mean(segment_errors)
        segment_mid_point = (bin_edges[i] + bin_edges[i+1]) / 2

        # Update maximum errors and midpoints for segments above and below x_star
        if segment_mid_point > x_star && segment_mean_error > max_error_above
            max_error_above = segment_mean_error
            mid_point_above = segment_mid_point
        end
        if segment_mid_point < x_star && segment_mean_error > max_error_below
            max_error_below = segment_mean_error
            mid_point_below = segment_mid_point
        end
    end

    # Return the midpoints for the highest error segments and errors for all segments
    return mid_point_below, mid_point_above, all_segment_errors
end

function find_max_segmented_errs(data::NN_Data, model::Chain, x_star::Vector{Float64}, n_segments::Int)
    
    # Concatenate training and testing data
    x = hcat(data.x_train, data.x_test)
    y = hcat(data.y_train, data.y_test)

    # Initialize vectors to store x_below and x_above
    x_belows = Float64[]
    x_aboves = Float64[]
    # Calculate errors for each sample
    errors = abs.(model(x) .- y)

    for i in 1:size(x, 1)
        x_below, x_above, _ = calculate_segmented_errs(x[i,:], errors, x_star[i], n_segments)
        push!(x_belows, x_below)
        push!(x_aboves, x_above)
    end

    return x_belows, x_aboves

end

# plot segmented errors 
function plot_segmented_errs(data::NN_Data, model::Chain, x_star::Vector{Float64}, n_segments::Int) 
    
    # Concatenate training and testing data
    x = hcat(data.x_train, data.x_test)
    y = hcat(data.y_train, data.y_test)

    # Calculate errors for each sample
    errors = abs.(model(x) .- y)

    # Create an empty array to hold all the individual plots
    plots_array = []

    for i in 1:size(x, 1)
        # Calculate midpoints, segment errors, and get all segment errors for the current variable
        mid_point_below, mid_point_above, all_segment_errors = calculate_segmented_errs(x[i, :], errors, x_star[i], n_segments)

        # Calculate the midpoints of each segment for plotting
        min_val, max_val = minimum(x[i, :]), maximum(x[i, :])
        bin_edges = range(min_val, stop = max_val, length = n_segments+1)
        segment_midpoints = [(bin_edges[j] + bin_edges[j+1]) / 2 for j in 1:n_segments]
        
        # Calculate mean errors for each segment where possible
        segment_mean_errors = Float32[isempty(segment) ? NaN : mean(segment) for segment in all_segment_errors]

        # Create the individual plot for this variable
        p = bar(segment_midpoints, segment_mean_errors, legend=false, title="Variable $i", xlabel="Segment Midpoint", ylabel="Mean Error", alpha=0.75)
        
        # Ensure x_star is included in the x-axis range
        extended_min_val = min(min_val, x_star[i]) - 1
        extended_max_val = max(max_val, x_star[i]) + 1
        xlims!(p, extended_min_val, extended_max_val)
        
        # Highlighting the x_star with a vertical line
        vline!(p, [x_star[i]], line=(:dash, 2), color=:red, label="x_star")

        # Add annotation for x_star at the top of the plot
        max_y_value = maximum(segment_mean_errors[.!isnan.(segment_mean_errors)]) * 1.05
        annotate!(p, [(x_star[i], max_y_value, "x_star")])

        # Marking mid_point_below and mid_point_above with scatter points
        scatter!(p, [mid_point_below], [max_y_value], color=:green, label="mid_point_below")
        scatter!(p, [mid_point_above], [max_y_value], color=:blue, label="mid_point_above")

        # Store the plot in the array
        push!(plots_array, p)
    end

    return plots_array

end

function generate_resample_config(sampling_config::Sampling_Config, x_star::Vector{Float64}, scale_factor::Float64, sample_size_method::Tuple{String, Float64},
    strategy::String; x_above::Vector{Float64} = Vector{Float64}(), x_below::Vector{Float64} = Vector{Float64}())

    
    lb_prev = sampling_config.lb
    ub_prev = sampling_config.ub
    interval_prev = sampling_config.ub - sampling_config.lb
    size_prev = sampling_config.n_samples
    density_prev = interval_prev / sampling_config.n_samples

    # strategy 1: fixed percentage of the search space
    if strategy == "fixed_percentage"
        sampling_config.lb = max.(x_star .- 0.5 * scale_factor * interval_prev, lb_prev)
        sampling_config.ub = min.(x_star .+ 0.5 * scale_factor * interval_prev, ub_prev)
    # strategy 2: error based resampling
    elseif strategy == "error_based"
        # x_above and x_below are provided for error_based strategy
        if isempty(x_above) || isempty(x_below)
            error("x_above and x_below must be provided for the error_based strategy.")
        end
        sampling_config.lb = max.(x_star .- scale_factor * abs.(x_star - x_below), lb_prev)
        sampling_config.ub = min.(x_star .+ scale_factor * abs.(x_star - x_above), ub_prev)
    # strategy 3: segmented error based
    elseif strategy == "segmented_error"
        # x_above and x_below are provided for segmented_error strategy
        if isempty(x_above) || isempty(x_below)
            error("x_above and x_below must be provided for the segmented_error strategy.")
        end
        sampling_config.lb = max.(x_star .- scale_factor * abs.(x_star - x_below), lb_prev)
        sampling_config.ub = min.(x_star .+ scale_factor * abs.(x_star - x_above), ub_prev)     
    else
        error("Invalid resampling strategy specified.")
    end

    # determine the number of samples based on the specified method
    if sample_size_method[1] == "fixed_percentage_density"
        sampling_config.n_samples = round(Int, maximum((sampling_config.ub - sampling_config.lb) ./ (sample_size_method[2] * density_prev)))
    elseif sample_size_method[1] == "fixed_percentage_size"
        sampling_config.n_samples = round(Int, sample_size_method[2] * size_prev)
    else
        error("Invalid sample size method specified.")
    end

    return sampling_config
end

# plot the learning curve based on the loss history
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