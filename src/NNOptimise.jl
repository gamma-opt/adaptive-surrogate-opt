using Flux
using Surrogates

include("../src/NNSurrogate.jl")
include("../src/NNJuMP.jl")

# evaluate the MIP solution

function gap_abs(f_hat::Float64, f::Float64)
    return abs(f_hat - f)/abs(f)
end

function solution_evaluate(MILP_model::Model, func::Function; mean = 0, std = 1)
    
    f_hat = objective_value(MILP_model)
    x_hat_norm = [value.(MILP_model[:x][0,i]) for i in 1:length(MILP_model[:x][0,:])]
    x_hat = [value.(MILP_model[:x][0,i]) for i in 1:length(MILP_model[:x][0,:])] .* std .+ mean
    f_true = func(Tuple(x_hat))
    gap = gap_abs(f_hat, f_true)
    
    println("        MIP solution: ", vec(x_hat))
    println("     Objective value: ", f_hat)
    println("True objective value: ", f_true)
    println("                 Gap: ", gap)

    if gap > 1e-2
        println("Warning: the MIP solution is not accurate enough!")
    end

    return f_hat, f_true, vec(x_hat), x_hat_norm, gap

end

# use generate_data to generate new samples around x^* and evaluate the function value at these samples

function resample(func::Function, x_star::Vector{Float64}, delta::Float64, n_samples::Int, sampling_method::Any)
    
    # generate new samples around x_star
    L_bounds = x_star .- delta
    U_bounds = x_star .+ delta
    data = generate_data(func, [L_bounds, U_bounds], n_samples, sampling_method, 0.8)
    
    return data, L_bounds, U_bounds
    
end

# retrain the surrogate model using the new data, considering freezing some layers parameters
function retrain_surrogate(data::NN_Data, config::NN_Config, model::Chain)
    
    # retrain the neural network model using the new samples, considering freezing the weights of the first few layers (c.freeze = 1)
    result = NN_train(data, config, trained_model = model)
    NN_results(config, result)
    
    # get the trained neural network model from the results
    model = result.model
    
    return model
    
end


# use the surrogate model to find the next x^*
function resolve_MILP(model::Chain, config::NN_Config, L_bounds::Vector{Float32}, U_bounds::Vector{Float32})
    
    L_bounds = vcat(Float32.(L_bounds), fill(Float32(-1e6), sum(config.layer[2:end])))
    U_bounds = vcat(Float32.(U_bounds), fill(Float32(1e6), sum(config.layer[2:end])))

    MILP_model = JuMP_Model(model, L_bounds, U_bounds)
    optimize!(MILP_model)

    return MILP_model
    
end




# compute the function value at x^* and check its precision with the true function value (if within a certain threshold, then stop)

# return the best x^* found so far