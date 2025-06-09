using KernelDensity
using Statistics
using Plots: plot, vline, hline

# get the predictive distribution
function predict_dist(data::NN_Data, model::Chain, pred_n::Int=100, top_n::Int=10)

    start_time = time()

    x = hcat(data.x_train, data.x_test)
    num_samples = size(x, 2)
    output_dim = size(model(x[:, 1]), 1)

    trainmode!(model)
    predictions = [model(x) for _ in 1:pred_n]
    predictions_per_sample = [zeros(Float64, pred_n, output_dim) for _ in 1:num_samples]
    
    # populate the predictions per sample
    for i in 1:num_samples
        for j in 1:pred_n
            predictions_per_sample[i][j, :] = predictions[j][:, i]
        end
    end

    # calculate mean and standard deviation for each dimension
    stds = std(predictions)
    computation_time = time() - start_time

    means = mean(predictions)
    

    # find indices of the points with the highest overall standard deviation
    overall_stds = mean(stds, dims=1)
    top_indices = sortperm(vec(overall_stds), rev=true)[1:top_n]
    top_points = x[:, top_indices]

    return predictions, predictions_per_sample, means, stds, top_points, computation_time
    
end

# get a point estimate (use the mean for this)
function predict_point(data::NN_Data, model::Chain, n::Int, selected_y_dim::Int=1)
    
    x = hcat(data.x_train, data.x_test)
    num_samples = size(x, 2)

    predictions = Array{Float64, 2}(undef, n, num_samples)

    # populate the predictions array for the selected dimension
    for i in 1:n
        prediction = model(x)
        predictions[i, :] = prediction[selected_y_dim, :]  # extract only the selected dimension
    end

    # Calculate mean and std along the predictions dimension
    means = mean(predictions, dims=1)
    stds = std(predictions, dims=1)

    # Reshape means and stds to be 1D vectors
    means = vec(means)
    stds = vec(stds)

    return means, stds
end

# plot the predictive distribution
function kdeplot(filtered_predictions_per_sample::Vector, mean_prediction::Union{Float32, Float64})

    # default kernal is Normal
    prediction_kde = kde(filtered_predictions_per_sample) 

    xs = prediction_kde.x
    densities = prediction_kde.density

    plot(xs, densities, fill=(0, 0.8, :cornflowerblue), label="Kernel density estimation", linewidth = 0)

    vline!([mean_prediction], color=:lightcoral, linestyle=:dash, linewidth=2, label="Mean")

end


function generate_resample_configs_mc(sampling_config::Sampling_Config, x_top_var::Matrix, scalar_radius::Float64, scalar_n_samples::Float64, mean::Matrix, std::Matrix)
    
    start_time = time()

    # Determine the number of parameters (dimensions) and points
    num_parameters, num_points = size(x_top_var)
    
    # Calculate the adjusted number of samples for each new config
    adjusted_n_samples = Int(ceil(sampling_config.n_samples * scalar_n_samples / num_points))
    
    # Initialise a vector to hold all the new sampling configurations
    # new_configs_norm = Vector{Sampling_Config}(undef, num_points)
    new_configs = Vector{Sampling_Config}(undef, num_points)
    
    # Loop through each high variance point to create new sampling configs
    for i in 1:num_points
        new_lb = Vector{Float64}(undef, num_parameters)
        new_ub = Vector{Float64}(undef, num_parameters)
        
        # Calculate new bounds by scaling the original bounds around the center points
        for j in 1:num_parameters
            center = x_top_var[j, i]
            radius = (sampling_config.ub[j] - sampling_config.lb[j]) * scalar_radius / 2
            new_lb[j] = max(sampling_config.lb[j], center - radius)
            new_ub[j] = min(sampling_config.ub[j], center + radius)
        end
        
        # Create a new Sampling_Config for the current high variance point
        # new_configs_norm[i] = Sampling_Config(adjusted_n_samples, new_lb, new_ub)
        new_configs[i] = Sampling_Config(adjusted_n_samples, new_lb .* vec(std) .+ vec(mean), new_ub .* vec(std) .+ vec(mean))     
    end

    # find the extreme config
    min_lb = copy(new_configs[1].lb)
    max_ub = copy(new_configs[1].ub)
    for config in new_configs
        # Element-wise comparison to find the minimum and maximum
        min_lb = min.(min_lb, config.lb)
        max_ub = max.(max_ub, config.ub)
    end
    n_samples = sum([config.n_samples for config in new_configs])
    new_config = Sampling_Config(n_samples, min_lb, max_ub)

    computation_time = time() - start_time

    return new_configs, new_config, computation_time
end



