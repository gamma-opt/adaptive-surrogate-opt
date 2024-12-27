# module NNSurrogate

using Plots
using GLMakie, CairoMakie
using Statistics, StatsBase
using Surrogates
using Flux
using Flux: params, train!
using LinearAlgebra
using LaTeXStrings


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

function load_data(file_path::String, splits::Float64, num_features::Int, num_outputs::Int)
    
    # get the location of the script file
    root = dirname(@__FILE__)

    # a robust representation of the filepath to data file
    csv_file_path = joinpath(root, file_path)

    # read the data from the csv file
    rdata = DataFrame(CSV.File(csv_file_path, header=false))

    # extract x (first 5 columns) and y (6th column)
    x = Matrix(rdata[:, 1:num_features])'
    y = reshape(rdata[:, num_features + num_outputs], :, num_outputs)'

    # split the data into train set and test set
    train_data, test_data = Flux.splitobs((x, y), at = splits)

    # convert the data to the format of NN_Data
    data_flame = NN_Data()
    data_flame.x_train = Float32.(train_data[1])
    data_flame.y_train = Float32.(train_data[2])
    data_flame.x_test = Float32.(test_data[1])
    data_flame.y_test = Float32.(test_data[2])

    return data_flame

end

function generate_and_combine_data(f::Function, configs::Vector{Sampling_Config}, sampling_method::Any, splits::Float64)
    
    combined_x_train, combined_y_train, combined_x_test, combined_y_test = [], [], [], []

    # initialise the data
    data = NN_Data()

    n = 0
    for (index, config) in enumerate(configs)
        data = generate_data(f, config, sampling_method, splits)
        println("Data generated for config: #", index)
        push!(combined_x_train, data.x_train)
        push!(combined_y_train, data.y_train)
        push!(combined_x_test, data.x_test)
        push!(combined_y_test, data.y_test)
    end

    # Concatenate the collected data
    data.x_train = hcat(combined_x_train...)
    data.y_train = hcat(combined_y_train...)
    data.x_test = hcat(combined_x_test...)
    data.y_test = hcat(combined_y_test...)

    return data
end

# sample data points from given dataset and track their original indices
function extract_data_from_given_dataset(x_not_selected, y_not_selected, configs::Vector{Sampling_Config}, complement_indices)

    results_x = []
    results_y = []
    sampled_indices_per_config = []

    # Filter x and y points based on Sampling_Config bounds
    for config in configs
        valid_indices = findall(i -> all(config.lb .<= x_not_selected[:, i] .<= config.ub), 1:size(x_not_selected, 2))

        # If enough valid data points exist, sample them
        if length(valid_indices) >= config.n_samples
            sampled_indices = StatsBase.sample(valid_indices, config.n_samples, replace = false)     
        else
            println("Not enough data points meet the criteria for config with bounds $(config.lb) to $(config.ub)")
            sampled_indices = valid_indices    
        end

        # Store the original indices of the sampled data points
        original_indices = complement_indices[sampled_indices]
        push!(sampled_indices_per_config, original_indices)
        push!(results_x, x_not_selected[:, sampled_indices])
        push!(results_y, y_not_selected[:, sampled_indices])

    end
       
    return results_x, results_y, vcat(sampled_indices_per_config...)

end

# filters the data based on the provided lower and upper bounds
function filter_data_within_bounds(data::NN_Data, lb_filter::Vector, ub_filter::Vector)
    
    # Function to check if all elements of x are within the specified bounds
    function within_bounds(x, lb, ub)
        all(lb .<= x .<= ub)
    end
    
    # Filter x_train and y_train
    train_indices = [within_bounds(data.x_train[:, i], lb_filter, ub_filter) for i in 1:size(data.x_train, 2)]
    data.x_train = data.x_train[:, train_indices]
    data.y_train = data.y_train[:, train_indices]

    # Filter x_test and y_test
    test_indices = [within_bounds(data.x_test[:, i], lb_filter, ub_filter) for i in 1:size(data.x_test, 2)]
    data.x_test = data.x_test[:, test_indices]
    data.y_test = data.y_test[:, test_indices]

    return data
end

# combine two datasets of the NN_Data type
function combine_datasets(data1::NN_Data, data2::NN_Data; splits::Float64=0.8)::NN_Data
    
    # Create a new NN_Data instance to hold the combined data
    data = NN_Data()
    
    # Combine both training and testing data from both datasets
    x = hcat(data1.x_train, data2.x_train, data1.x_test, data2.x_test)
    y = hcat(data1.y_train, data2.y_train, data1.y_test, data2.y_test)

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
function normalise_data(data::NN_Data, norm_y::Bool = false) 
    
    data_norm = NN_Data()
    μ, σ = mean(data.x_train, dims=2), std(data.x_train, dims=2)  # compute the mean and standard deviation of the training set
    data_norm.x_train = Flux.normalise(data.x_train)  # normalise the training set
    data_norm.x_test = (data.x_test .- μ) ./ σ  # normalise the test set using the mean and standard deviation of the training set
    
    if norm_y
        μ_y, σ_y = mean(data.y_train, dims=2), std(data.y_train, dims=2)  # compute the mean and standard deviation of the training set
        data_norm.y_train = Flux.normalise(data.y_train)  # normalise the training set
        data_norm.y_test = (data.y_test .- μ_y) ./ σ_y  # normalise the test set using the mean and standard deviation of the training set
        return data_norm, μ, σ, μ_y, σ_y
    else
        data_norm.y_train = data.y_train
        data_norm.y_test = data.y_test
        return data_norm, μ, σ
    end
   
end

# define the loss function with L2 regularization
# pure mse if lambda = 0
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
    # loss(x,y) = Flux.mse(result.model(x), y) 
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
                if epoch % 100 == 0
                    println("Epoch: $epoch, Fold: $(length(train_err_fold)+1)")
                end
            end
            push!(train_err_fold, train_err_epoch[end])           
        end 
           
        result.err_hist = vec(mean(reshape(train_err_epoch, c.k, c.epochs), dims = 1))
        # result.train_err = [mean(train_err_fold), loss_rrmse(data.x_train, data.y_train, result.model), loss_mape(data.x_train, data.y_train, result.model)]
        result.train_err = [Flux.mse(result.model(data.x_train), data.y_train), loss_rrmse(data.x_train, data.y_train, result.model), loss_mape(data.x_train, data.y_train, result.model)]
        
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
            if epoch % 100 == 0
                println("Epoch: $epoch")
            end                                  
        end
        
        result.err_hist = train_err_epoch
        # result.train_err = [train_err_epoch[end], loss_rrmse(data.x_train, data.y_train, result.model), mean(abs.((result.model(data.x_train) .- data.y_train) ./ data.y_train)) ]
        result.train_err = [Flux.mse(result.model(data.x_train), data.y_train), loss_rrmse(data.x_train, data.y_train, result.model), mean(abs.((result.model(data.x_train) .- data.y_train) ./ data.y_train))]
        
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

# find the point with the maximum error below and above x_star
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


# visualisation

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

# visualise the surrogate model using tricontour plot, marking other points worth mentioning
function plot_dual_contours(data::NN_Data, model::Chain, x_star::Vector{Float64}, scattered_point_label::String, scattered_point, selected_dim_x::Vector{Int}, selected_dim_y::Int=1)
    
    # extract the selected dimensions
    x = hcat(data.x_train, data.x_test)
    x1 = collect(x[selected_dim_x[1], :])
    x2 = collect(x[selected_dim_x[2], :])

    # create a combined matrix of x1 and x2, then find unique rows and their indices
    points = [x1 x2]
    unique_ind = unique(i -> points[i, :], 1:size(points, 1))
    x1_unique = x1[unique_ind]
    x2_unique = x2[unique_ind]

    # extract the output
    y_true = collect(vec(hcat(data.y_train, data.y_test)[selected_dim_y, :]))
    y_pred = collect(vec(model(x)[selected_dim_y, :]))

    # Select y_true and y_pred based on unique indices
    y_true_unique = y_true[unique_ind]
    y_pred_unique = y_pred[unique_ind]

    # Determine color limits based on combined data
    global_min = min(minimum(y_true_unique), minimum(y_pred_unique))
    global_max = max(maximum(y_true_unique), maximum(y_pred_unique))
    levels = range(global_min, stop = global_max, length = 11)  # 10 intervals

    # plot the tricontour of the true output and the surrogate output
    fig = Figure(size = (750, 400))
    # fig = Figure(size = (800, 400))  # for the blade design problem

    ax1 = Axis(fig[1, 1], xlabel = "x₁",
    ylabel = "x₂", title = "Simulator",xlabelsize = 15, ylabelsize = 15, xlabelfont="Arial", ylabelfont="Arial")
    
    ax2 = Axis(fig[1, 3], xlabel = "x₁",
    ylabel = "x₂", title = "Surrogate", xlabelsize = 15, ylabelsize = 15, xlabelfont="Arial", ylabelfont="Arial")
    
    # plot the tricontour of the true output
    tr_true = tricontourf!(ax1, x1_unique, x2_unique, y_true_unique, levels = levels, triangulation = Makie.DelaunayTriangulation(), colormap = :viridis)
    # Makie.scatter!(ax1, x_star[selected_dim_x[1]], x_star[selected_dim_x[2]], color = y_true)
    sca = Makie.scatter!(ax1, x_star[selected_dim_x[1]], x_star[selected_dim_x[2]], color = :green)
    # text!(ax1, x_star[selected_dim_x[1]], x_star[selected_dim_x[2]], text = "x_star", align = (:center, :top))
    
    # plot the tricontour of the surrogate output
    tr_pred = tricontourf!(ax2, x1_unique, x2_unique, y_pred_unique, levels = levels, triangulation = Makie.DelaunayTriangulation(), colormap = :viridis)
    sca = Makie.scatter!(ax2, x_star[selected_dim_x[1]], x_star[selected_dim_x[2]], color = :green)
    
    # plot the scattered points
    sca2 = Makie.scatter!(NaN, NaN)
    start_index = scattered_point_label == "sub-optimal solutions" ? 2 : 1
    # in case Solution count = 1
    if scattered_point_label == "current optimum" && size(scattered_point, 1) == 1
        Legend(fig[2, :], [sca], [L"\hat{x}^*"], orientation = :horizontal, labelsize = 15)
    else
        for val in scattered_point[start_index:end]
            sca2 = Makie.scatter!(ax1, val[selected_dim_x[1]], val[selected_dim_x[2]], color = :orange)
            sca2 = Makie.scatter!(ax2, val[selected_dim_x[1]], val[selected_dim_x[2]], color = :orange)
        end
        Legend(fig[2, :], [sca, sca2], ["current optimum", scattered_point_label], orientation = :horizontal, labelsize = 15)

    end
    
    Colorbar(fig[1, 2], tr_true)
    Colorbar(fig[1, 4], tr_pred)

    return fig

end


# visualise the scattered data points using tricontour plot
function plot_single_contour(data::NN_Data, model::Chain, x_star::Vector{Float64}, label::String, results::Vector, scattered_point_label::String, scattered_point, selected_dim::Vector{Int})
    
    # extract the selected dimensions
    x = hcat(data.x_train, data.x_test)
    x1 = collect(x[selected_dim[1], :])
    x2 = collect(x[selected_dim[2], :])

    # create a combined matrix of x1 and x2, then find unique rows and their indices
    points = [x1 x2]
    unique_ind = unique(i -> points[i, :], 1:size(points, 1))
    x1_unique = x1[unique_ind]
    x2_unique = x2[unique_ind]

    # extract the output
    y = results[unique_ind]

    # plot the tricontour of the true output and the surrogate output
    fig = Figure(size = (400, 410))
    ax1 = Axis(fig[1, 1], xlabel = "x₁", ylabel = "x₂", title = label)
    sca1 = Makie.scatter!(ax1, x_star[selected_dim[1]], x_star[selected_dim[2]], color = :green)
    # plot the tricontour of the true output
    tr = tricontourf!(ax1, x1_unique, x2_unique, y, triangulation = Makie.DelaunayTriangulation(), colormap = :viridis)
    
    # plot the scattered points
    sca2 = Makie.scatter!(NaN, NaN)
    for val in scattered_point[1:end]
        sca2 = Makie.scatter!(ax1, val[selected_dim[1]], val[selected_dim[2]], color = :orange)
    end
    # # determine the color limits for the plots
    # min_val = minimum([y])
    # max_val = maximum([y])
    # color_limits = (min_val, max_val)
    
    # # Set the color limits for the tricontour plots
    # tr.climits = color_limits
    
    Colorbar(fig[1, 2], tr)
    Legend(fig[2, :], [sca1, sca2], ["current optimum", scattered_point_label], orientation = :horizontal, labelsize = 15)

    return fig

end

# end # module