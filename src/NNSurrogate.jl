using Plots
using CairoMakie
using Statistics, StatsBase
using Surrogates
using Flux
using Flux: params, train!
using LinearAlgebra
using LaTeXStrings
using DataFrames, CSV

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

function create_nn_data(x, y; splits = 0.8)
    data = NN_Data()
    train_data, test_data = Flux.splitobs((x, y), at = splits)
    
    data.x_train = Float32.(train_data[1])
    data.y_train = Float32.(train_data[2])
    data.x_test = Float32.(test_data[1])
    data.y_test = Float32.(test_data[2])
    
    return data
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

    all_sampled_indices = []
    start_time = time()

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

        # Collect all sampled indices
        append!(all_sampled_indices, sampled_indices)

    end

    # Get unique indices and extract corresponding data
    unique_sampled_indices = unique(all_sampled_indices)
    results_x = x_not_selected[:, unique_sampled_indices]
    results_y = y_not_selected[:, unique_sampled_indices]

    selected_indices = complement_indices[unique_sampled_indices]

    println("Number of data points added: $(length(selected_indices))")
    
    complement_indices = setdiff(complement_indices, selected_indices)
    computation_time = time() - start_time
       
    return results_x, results_y, selected_indices, complement_indices, computation_time

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
function plot_dual_contours(data::NN_Data, model::Chain, x_star::Vector{Float64}, scattered_point_label::String, scattered_point, selected_dim_x::Vector{Int}, selected_dim_y::Int=1, global_optimum::Union{Nothing, Vector{Float64}}=nothing)
    
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
    sca = Makie.scatter!(ax1, x_star[selected_dim_x[1]], x_star[selected_dim_x[2]], color = :lightblue)
    # text!(ax1, x_star[selected_dim_x[1]], x_star[selected_dim_x[2]], text = "x_star", align = (:center, :top))
    
    # plot the tricontour of the surrogate output
    tr_pred = tricontourf!(ax2, x1_unique, x2_unique, y_pred_unique, levels = levels, triangulation = Makie.DelaunayTriangulation(), colormap = :viridis)
    sca = Makie.scatter!(ax2, x_star[selected_dim_x[1]], x_star[selected_dim_x[2]], color = :lightblue)
    
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

        if !isnothing(global_optimum)
            sca3 = Makie.scatter!(ax1, global_optimum[selected_dim_x[1]], global_optimum[selected_dim_x[2]], color = :red)
            sca3 = Makie.scatter!(ax2, global_optimum[selected_dim_x[1]], global_optimum[selected_dim_x[2]], color = :red)
            Legend(fig[2, :], [sca, sca2, sca3], ["current optimum", scattered_point_label, "global optimum"], orientation = :horizontal, labelsize = 15)
        else
            Legend(fig[2, :], [sca, sca2], ["current optimum", scattered_point_label], orientation = :horizontal, labelsize = 15)
        end

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

# end 