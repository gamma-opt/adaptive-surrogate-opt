using KernelDensity
using Statistics
using Plots: plot, vline, hline
using Distributions  # Required for Normal distribution functions

# get the predictive distribution
function predict_dist(data::NN_Data, model::Chain, pred_n::Int=100, top_n::Int=10)

    x = hcat(data.x_train, data.x_test)

    num_samples = size(x, 2)

    trainmode!(model)
    predictions = [model(x) for _ in 1:pred_n]
    predictions_per_sample = [zeros(pred_n) for _ in 1:num_samples]
    
    for i in 1:num_samples
        predictions_per_sample[i] = vec(getindex.(predictions, i))
    end

    means = mean(predictions)

    # find the points with the highest variance
    stds = std(predictions)
    top_indices = sortperm(stds[1, :], rev=true)[1:top_n]
    top_points = x[:, top_indices]

    return predictions_per_sample, means, stds, top_points
    
end

# # get a point estimate (use the mean for this)
function predict_point(data::NN_Data, model::Chain, n::Int)

    x = hcat(data.x_train, data.x_test)

    predictions = [model(x) for _ in 1:n]
    
    return mean(predictions), std(predictions)
    
end

# handle the outliner in the predictions using the Interquartile Range (IQR)method
function remove_outliers_per_dist(predictions_per_sample::Vector)

    Q1 = quantile(predictions_per_sample, 0.25)
    Q3 = quantile(predictions_per_sample, 0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    # remove the outliers
    filtered_predictions_per_sample = [x for x in predictions_per_sample if (x >= lower_bound) && (x <= upper_bound)]

    return filtered_predictions_per_sample
    
end

function remove_outliers(data::NN_Data, predictions::Vector)

    num_samples = size(data.x_test, 2)
    filtered_predictions = [remove_outliers_per_dist(predictions[i]) for i in 1:num_samples]
    
    return filtered_predictions
    
end

# plot the predictive distribution
function kdeplot(filtered_predictions_per_sample::Vector, mean_prediction::Union{Float32, Float64})

    prediction_kde = kde(filtered_predictions_per_sample) 

    xs = prediction_kde.x
    densities = prediction_kde.density

    plot(xs, densities, fill=(0, 0.8, :cornflowerblue), label="Kernel density estimation", linewidth = 0)

    vline!([mean_prediction], color=:lightcoral, linestyle=:dash, linewidth=2, label="Mean")

end

# calculate the expected prediction errors
function expected_prediction_error(data::NN_Data, model::Chain, pred_n::Int=100, top_n::Int=10)

    x = hcat(data.x_train, data.x_test)
    y = hcat(data.y_train, data.y_test)

    num_samples = size(x, 2)

    predictions, _, _, _ = predict_dist(data, model, pred_n)
    prediction_errs = [zeros(pred_n) for _ in 1:num_samples]

    for i in 1:num_samples
        prediction_errs[i] = (abs.(predictions[i] .- y[i])).^2
    end

    # find the point with the highest ee
    means = mean.(prediction_errs)
    top_indices = sortperm(means, rev=true)[1:top_n]
    top_points = x[:, top_indices]

    return means, top_points
    
end

# find the max expected error
function find_max_ee(data::NN_Data, model::Chain, x_star::Vector{Float64}, n::Int)
    
    x = data.x_test

    # Calculate the errors (absolute difference)
    errors = expected_prediction_error(data::NN_Data, model::Chain, n::Int)

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

"""
# Parameters:
- μ_x: Predicted mean of f(x) by the Gaussian Process model.
- σ_x: Predicted standard deviation of f(x) by the Gaussian Process model.
- f_x_plus: Best observed value of the target function so far.
- xi: Exploration-exploitation trade-off parameter (default is 0.01).
"""

function calculate_ei(μ_x::Union{Float32, Float64}, σ_x::Union{Float32, Float64}, f_x_star::Union{Float32, Float64}, is_minimisation::Bool=true, xi::Float64=0.01) :: Float64
    
    if σ_x > 0.0
        # For minimisation, we want the EI to be high for predictions below f_x_star
        # For maximisation, we want the EI to be high for predictions above f_x_star
        improvement = is_minimisation ? (f_x_star - μ_x - xi) : (μ_x - f_x_star - xi)
        Z = improvement / σ_x
        return (f_x_star - μ_x - xi) * cdf(Normal(0, 1), Z) + σ_x * pdf(Normal(0, 1), Z)
    else
        return 0.0  # If σ_x is 0, the improvement is considered to be 0 as there is no uncertainty.
    end

end

# calculate the expected improvement
function expected_improvement(data::NN_Data, model::Chain, x_star::Vector{Float64}, pred_n::Int=100, top_n::Int=10, is_minimisation::Bool=true, xi::Float64=0.01)
 
    x = hcat(data.x_train, data.x_test)

    num_samples = size(x, 2)

    μ_x, σ_x = predict_point(data, model, pred_n)
    f_x_star = model(x_star)[1]

    expected_improvements = Float64[]

    for i in 1:num_samples
        push!(expected_improvements, calculate_ei(μ_x[i], σ_x[i], f_x_star, is_minimisation, xi))
    end

    # expected_prediction_error = mean(prediction_errs)
    indices = sortperm(expected_improvements, rev=true)[1:top_n]
    top_points = x[:, indices]

    return expected_improvements, top_points
    
end

# find the max expected improvement
function find_max_ei(data::NN_Data, model::Chain, x_star::Vector{Float64}, n::Int)
    
    x = data.x_test

    # Calculate the errors (absolute difference)
    errors = expected_improvement(data::NN_Data, model::Chain, x_star::Vector{Float64}, n::Int)

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

function generate_resample_configs_mc(sampling_config::Sampling_Config, x_top_var::Matrix, scalar_radius::Float64, scalar_n_samples::Float64, mean::Matrix, std::Matrix)
    
    # Determine the number of parameters (dimensions) and points
    num_parameters, num_points = size(x_top_var)
    
    # Calculate the adjusted number of samples for each new config
    adjusted_n_samples = Int(ceil(sampling_config.n_samples * scalar_n_samples / num_points))
    
    # Initialise a vector to hold all the new sampling configurations
    new_configs_norm = Vector{Sampling_Config}(undef, num_points)
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
        new_configs_norm[i] = Sampling_Config(adjusted_n_samples, new_lb, new_ub)
        new_configs[i] = Sampling_Config(adjusted_n_samples, new_lb .* vec(std) .+ vec(mean), new_ub .* vec(std) .+ vec(mean))
    end
    
    return new_configs_norm, new_configs
end


