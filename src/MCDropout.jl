using KernelDensity
using Statistics
using Plots
using Distributions  # Required for Normal distribution functions

# get the predictive distribution
function predict_dist(data::NN_Data, model::Chain, n::Int)

    num_samples = size(data.x_test, 2)

    trainmode!(model)
    predictions = [model(data.x_test) for _ in 1:n]
    predictions_per_sample = [zeros(n) for _ in 1:num_samples]
    
    for i in 1:num_samples
        predictions_per_sample[i] = vec(getindex.(predictions, i))
    end

    return predictions_per_sample
    
end

# get a point estimate (use the mean for this)
function predict_point(data::NN_Data, model::Chain, n::Int)

    predictions = [model(data.x_test) for _ in 1:n]
    
    return mean(predictions), std(predictions)
    
end

# handle the outliner in the predictions using the  Interquartile Range (IQR)method
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

    plot(xs, densities, fill=(0, .5, :blue), label="Kernel density estimation", linewidth = 1)

    vline!([mean_prediction], color=:red, linestyle=:dash, linewidth=1, label="Mean")

end

# calculate the expected prediction errors
function expected_prediction_error(data::NN_Data, model::Chain, n::Int)

    num_samples = size(data.x_test, 2)

    predictions = predict_dist(data, model, n)
    prediction_errs = [zeros(n) for _ in 1:num_samples]

    for i in 1:num_samples
        prediction_errs[i] = abs.(predictions[i] .- data.y_test[i])
    end

    # expected_prediction_error = mean(prediction_errs)

    return mean.(prediction_errs)
    
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

function calculate_ei(μ_x::Union{Float32, Float64}, σ_x::Union{Float32, Float64}, f_x_star::Union{Float32, Float64}, xi::Float64=0.01) :: Float64
    if σ_x > 0.0
        Z = (μ_x - f_x_star - xi) / σ_x
        return (μ_x - f_x_star - xi) * cdf(Normal(0, 1), Z) + σ_x * pdf(Normal(0, 1), Z)
    else
        return 0.0  # If σ_x is 0, the improvement is considered to be 0 as there is no uncertainty.
    end
end

# calculate the expected improvement
function expected_improvement(data::NN_Data, model::Chain, x_star::Vector{Float64}, n::Int)
 
    num_samples = size(data.x_test, 2)

    predictions = predict_dist(data, model, n)
    μ_x, σ_x = predict_point(data, model, n)
    f_x_star = model(x_star)[1]

    expected_improvements = Float64[]

    for i in 1:num_samples
        push!(expected_improvements, calculate_ei(μ_x[i], σ_x[i], f_x_star, 0.01))
    end

    # expected_prediction_error = mean(prediction_errs)

    return expected_improvements
    
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



