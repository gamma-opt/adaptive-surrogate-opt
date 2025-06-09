# duplicated from Gogeta.jl (https://github.com/gamma-opt/Gogeta.jl), with modifications on the number of layers

"""
    function NN_formulate!(jump_model::JuMP.Model, NN_model::Flux.Chain, U_in, L_in; U_bounds=nothing, L_bounds=nothing, U_out=nothing, L_out=nothing, bound_tightening="fast", compress=false, parallel=false, silent=true)

Creates a mixed-integer optimization problem from a `Flux.Chain` model.

The parameters are used to specify what kind of bound tightening and compression will be used.

A dummy objective function of 1 is added to the model. The objective is left for the user to define.

# Arguments
- `jump_model`: The constraints and variables will be saved to this optimization model.
- `NN_model`: Neural network model to be formulated.
- `U_in`: Upper bounds for the input variables.
- `L_in`: Lower bounds for the input variables.

# Optional arguments
- `bound_tightening`: Mode selection: "fast", "standard", "output" or "precomputed"
- `compress`: Should the model be simultaneously compressed?
- `parallel`: Runs bound tightening in parallel. `set_solver!`-function must be defined in the global scope, see documentation or examples.
- `U_bounds`: Upper bounds. Needed if bound_tightening="precomputed"
- `L_bounds`: Lower bounds. Needed if bound_tightening="precomputed"
- `U_out`: Upper bounds for the output variables. Needed if bound_tightening="output".
- `L_out`: Lower bounds for the output variables. Needed if bound_tightening="output".
- `silent`: Controls console ouput.

"""
function NN_formulate!(jump_model::JuMP.Model, NN_model::Flux.Chain, U_in, L_in; bound_tightening="fast", compress=false, parallel=false, U_bounds=nothing, L_bounds=nothing, U_out=nothing, L_out=nothing, silent=true)

    oldstdout = stdout
    if silent redirect_stdout(devnull) end

    if compress
        println("Starting compression...")
    else
        println("Creating JuMP model...")
    end

    empty!(jump_model)
    @assert bound_tightening in ("precomputed", "fast", "standard", "output") "Accepted bound tightening modes are: precomputed, fast, standard, output."

    if bound_tightening == "precomputed" @assert !isnothing(U_bounds) && !isnothing(L_bounds) "Precomputed bounds must be provided." end

    if bound_tightening == "precomputed"
        K = length(NN_model) # number of layers (input layer not included)
    else
        K = (length(NN_model) - 1) ÷ 2 # number of layers (excluding input and dorpout layers)
    end
    
    W = deepcopy([Flux.params(NN_model)[2*k-1] for k in 1:K])
    b = deepcopy([Flux.params(NN_model)[2*k] for k in 1:K])

    # @assert all([NN_model[i].σ == relu for i in 1:K-1]) "Neural network must use the relu activation function."
    # @assert NN_model[K].σ == identity "Neural network must use the identity function for the output layer."
    
    removed_neurons = Vector{Vector}(undef, K)
    [removed_neurons[layer] = Vector{Int}() for layer in 1:K]

    input_length = Int((length(W[1]) / length(b[1])))
    neuron_count = [length(b[k]) for k in eachindex(b)]
    neurons(layer) = layer == 0 ? [i for i in 1:input_length] : [i for i in setdiff(1:neuron_count[layer], removed_neurons[layer])]
    @assert input_length == length(U_in) == length(L_in) "Initial bounds arrays must be the same length as the input layer"
        
    @variable(jump_model, x[layer = 0:K, neurons(layer)])
    @variable(jump_model, s[layer = 1:K-1, neurons(layer)])
    @variable(jump_model, z[layer = 1:K-1, neurons(layer)])
        
    @constraint(jump_model, [j = 1:input_length], x[0, j] <= U_in[j])
    @constraint(jump_model, [j = 1:input_length], x[0, j] >= L_in[j])
    
    if bound_tightening != "precomputed" 
        U_bounds = Vector{Vector}(undef, K)
        L_bounds = Vector{Vector}(undef, K)
    end
    
    # upper bound and lower bound constraints for output bound tightening
    ucons = Vector{Vector{ConstraintRef}}(undef, K)
    lcons = Vector{Vector{ConstraintRef}}(undef, K)

    [ucons[layer] = Vector{ConstraintRef}(undef, neuron_count[layer]) for layer in 1:K]
    [lcons[layer] = Vector{ConstraintRef}(undef, neuron_count[layer]) for layer in 1:K]

    layers_removed = 0 # how many strictly preceding layers have been removed at current loop iteration 

    for layer in 1:K # hidden layers and bounds for output layer

        println("\nLAYER $layer")

        if bound_tightening != "precomputed"

            # compute loose bounds
            if layer - layers_removed == 1
                U_bounds[layer] = [sum(max(W[layer][neuron, previous] * U_in[previous], W[layer][neuron, previous] * L_in[previous]) for previous in neurons(layer-1-layers_removed)) + b[layer][neuron] for neuron in neurons(layer)]
                L_bounds[layer] = [sum(min(W[layer][neuron, previous] * U_in[previous], W[layer][neuron, previous] * L_in[previous]) for previous in neurons(layer-1-layers_removed)) + b[layer][neuron] for neuron in neurons(layer)]
            else
                U_bounds[layer] = [sum(max(W[layer][neuron, previous] * max(0, U_bounds[layer-1-layers_removed][previous]), W[layer][neuron, previous] * max(0, L_bounds[layer-1-layers_removed][previous])) for previous in neurons(layer-1-layers_removed)) + b[layer][neuron] for neuron in neurons(layer)]
                L_bounds[layer] = [sum(min(W[layer][neuron, previous] * max(0, U_bounds[layer-1-layers_removed][previous]), W[layer][neuron, previous] * max(0, L_bounds[layer-1-layers_removed][previous])) for previous in neurons(layer-1-layers_removed)) + b[layer][neuron] for neuron in neurons(layer)]
            end

            # compute tighter bounds
            if bound_tightening == "standard"
                bounds = if parallel == true # multiprocessing enabled
                    pmap(neuron -> calculate_bounds(copy_model(jump_model), layer, neuron, W, b, neurons; layers_removed), neurons(layer))
                else
                    map(neuron -> calculate_bounds(jump_model, layer, neuron, W, b, neurons; layers_removed), neurons(layer))
                end
                # only change if bound is improved
                U_bounds[layer] = min.(U_bounds[layer], [bound[1] for bound in bounds])
                L_bounds[layer] = max.(L_bounds[layer], [bound[2] for bound in bounds])
            end
        end

        # output bounds calculated but no more constraints added
        if layer == K
            break
        end

        if compress 
            layers_removed = prune!(W, b, removed_neurons, layers_removed, neuron_count, layer, U_bounds, L_bounds) 
        end

        for neuron in neurons(layer)
            @constraint(jump_model, x[layer, neuron] >= 0)
            @constraint(jump_model, s[layer, neuron] >= 0)
            set_binary(z[layer, neuron])

            ucons[layer][neuron] = @constraint(jump_model, x[layer, neuron] <= max(0, U_bounds[layer][neuron]) * (1 - z[layer, neuron]))
            lcons[layer][neuron] = @constraint(jump_model, s[layer, neuron] <= max(0, -L_bounds[layer][neuron]) * z[layer, neuron])

            @constraint(jump_model, x[layer, neuron] - s[layer, neuron] == b[layer][neuron] + sum(W[layer][neuron, i] * x[layer-1-layers_removed, i] for i in neurons(layer-1-layers_removed)))
        end

        if length(neurons(layer)) > 0
            layers_removed = 0
        end 

    end

    # output layer
    @constraint(jump_model, [neuron in neurons(K)], x[K, neuron] == b[K][neuron] + sum(W[K][neuron, i] * x[K-1-layers_removed, i] for i in neurons(K-1-layers_removed)))

    # using output bounds in bound tightening
    if bound_tightening == "output"
        @assert length(L_out) == length(U_out) == neuron_count[K] "Incorrect length of output bounds array."

        println("Starting bound tightening based on output bounds as well as input bounds.")

        @constraint(jump_model, [neuron in 1:neuron_count[K]], x[K, neuron] >= L_out[neuron])
        @constraint(jump_model, [neuron in 1:neuron_count[K]], x[K, neuron] <= U_out[neuron])

        for layer in 1:K-1

            println("\nLAYER $layer")

            bounds = if parallel == true # multiprocessing enabled
                pmap(neuron -> calculate_bounds(copy_model(jump_model), layer, neuron, W, b, neurons; layers_removed), neurons(layer))
            else
                map(neuron -> calculate_bounds(jump_model, layer, neuron, W, b, neurons; layers_removed), neurons(layer))
            end

            # only change if bound is improved
            U_bounds[layer] = min.(U_bounds[layer], [bound[1] for bound in bounds])
            L_bounds[layer] = max.(L_bounds[layer], [bound[2] for bound in bounds])

            for neuron in neuron_count[layer]

                delete(jump_model, ucons[layer][neuron])
                delete(jump_model, lcons[layer][neuron])

                @constraint(jump_model, x[layer, neuron] <= max(0, U_bounds[layer][neuron]) * (1 - z[layer, neuron]))
                @constraint(jump_model, s[layer, neuron] <= max(0, -L_bounds[layer][neuron]) * z[layer, neuron])
            end

        end

        U_bounds[K] = U_out
        L_bounds[K] = L_out
    end

    @objective(jump_model, Max, 1)

    redirect_stdout(oldstdout)
    
    if compress
        new_model = build_model!(W, b, K, neurons)

        if bound_tightening != "precomputed"

            U_compressed = [U_bounds[layer][neurons(layer)] for layer in 1:K]
            filter!(neurons -> length(neurons) != 0, U_compressed)

            L_compressed = [L_bounds[layer][neurons(layer)] for layer in 1:K]
            filter!(neurons -> length(neurons) != 0, L_compressed)

            empty!(jump_model)
            NN_formulate!(jump_model, new_model, U_in, L_in; U_bounds=U_compressed, L_bounds=L_compressed, bound_tightening="precomputed", silent)

            return new_model, removed_neurons, U_compressed, L_compressed
        else
            return new_model, removed_neurons
        end
    else
        if bound_tightening != "precomputed"
            return U_bounds, L_bounds
        end
    end

end

"""
    function forward_pass!(jump_model::JuMP.Model, input)

Calculates the output of a JuMP model representing a neural network.
"""
function forward_pass!(jump_model::JuMP.Model, input)
    
    @assert length(input) == length(jump_model[:x][0, :]) "Incorrect input length."
    [fix(jump_model[:x][0, i], input[i], force=true) for i in eachindex(input)]
    
    try
        optimize!(jump_model)
        (last_layer, outputs) = maximum(keys(jump_model[:x].data))
        result = value.(jump_model[:x][last_layer, :])
        return [result[i] for i in 1:outputs]
    catch e
        println("ERROR: $e")
        @warn "Input or output outside of bounds or incorrectly constructed model."
        return [NaN]
    end

end


"""
    function copy_model(input_model, solver_params)

Creates a copy of a JuMP model. Solver has to be specified for each new copy. Used for parallelization.
"""
function copy_model(input_model)
    model = copy(input_model)
    try
        Main.set_solver!(model)
    catch e
        println(e)
        error("To use multiprocessing, 'set_solver!'-function must be correctly defined in the global scope for each worker process.")
    end
    return model
end

"""
    function calculate_bounds(model::JuMP.Model, layer, neuron, W, b, neurons; layers_removed=0)

Calculates the upper and lower activation bounds for a neuron in a ReLU-activated neural network.
"""
function calculate_bounds(model::JuMP.Model, layer, neuron, W, b, neurons; layers_removed=0)

    @objective(model, Max, b[layer][neuron] + sum(W[layer][neuron, i] * model[:x][layer-1-layers_removed, i] for i in neurons(layer-1-layers_removed)))
    optimize!(model)
    
    upper_bound = if termination_status(model) == OPTIMAL
        objective_value(model)
    else
        @warn "Layer $layer, neuron $neuron could not be solved to optimality."
        objective_bound(model)
    end

    set_objective_sense(model, MIN_SENSE)
    optimize!(model)
 
    lower_bound = if termination_status(model) == OPTIMAL
        objective_value(model)
    else
        @warn "Layer $layer, neuron $neuron could not be solved to optimality."
        objective_bound(model)
    end

    println("Neuron: $neuron, bounds: [$lower_bound, $upper_bound]")

    return upper_bound, lower_bound
end


"""
    function NN_compress(NN_model::Flux.Chain, U_in, L_in, U_bounds, L_bounds)

Compresses a neural network using precomputed bounds.

# Arguments
- `NN_model`: Neural network to be compressed.
- `U_in`: Upper bounds for the input variables.
- `L_in`: Lower bounds for the input variables.
- `U_bounds`: Upper bounds for the other neurons.
- `L_bounds`: Lower bounds for the other neurons.

Returns a `Flux.Chain` model of the compressed neural network.
"""
function NN_compress(NN_model::Flux.Chain, U_in, L_in, U_bounds, L_bounds)

    K = length(NN_model) # number of layers (input layer not included)
    # K = (length(NN_model) - 1) ÷ 2
    W = deepcopy([Flux.params(NN_model)[2*k-1] for k in 1:K])
    b = deepcopy([Flux.params(NN_model)[2*k] for k in 1:K])

    # @assert all([NN_model[i].σ == relu for i in 1:K-1]) "Neural network must use the relu activation function."
    # @assert NN_model[K].σ == identity "Neural network must use the identity function for the output layer."
    
    removed_neurons = Vector{Vector}(undef, K)
    [removed_neurons[layer] = Vector{Int}() for layer in 1:K]

    input_length = Int((length(W[1]) / length(b[1])))
    neuron_count = [length(b[k]) for k in eachindex(b)]
    neurons(layer) = layer == 0 ? [i for i in 1:input_length] : [i for i in setdiff(1:neuron_count[layer], removed_neurons[layer])]
    
    @assert input_length == length(U_in) == length(L_in) "Initial bounds arrays must be the same length as the input layer"

    layers_removed = 0

    for layer in 1:K-1
        layers_removed = prune!(W, b, removed_neurons, layers_removed, neuron_count, layer, U_bounds, L_bounds)

        if length(neurons(layer)) > 0
            layers_removed = 0
        end 

    end
    
    new_model = build_model!(W, b, K, neurons)
    return new_model, removed_neurons
end

"""
    function prune!(W, b, removed_neurons, layers_removed, neuron_count, layer, bounds_U, bounds_L)

Removes stabily active or inactive neurons in a network by updating the weights and the biases and the removed neurons list accordingly.
"""
function prune!(W, b, removed_neurons, layers_removed, neuron_count, layer, bounds_U, bounds_L)

    stable_units = Set{Int}() # indices of stable neurons
    unstable_units = false

    for neuron in 1:neuron_count[layer]

        if bounds_U[layer][neuron] <= 0 || iszero(W[layer][neuron, :]) # stabily inactive

            if neuron < neuron_count[layer] || length(stable_units) > 0 || unstable_units == true
                
                if iszero(W[layer][neuron, :]) && b[layer][neuron] > 0
                    for neuron_next in 1:neuron_count[layer+1] # adjust biases
                        b[layer+1][neuron_next] += W[layer+1][neuron_next, neuron] * b[layer][neuron]
                    end
                end

                push!(removed_neurons[layer], neuron)
            end

        elseif bounds_L[layer][neuron] >= 0 # stabily active
            
            if rank(W[layer][collect(union(stable_units, neuron)), :]) > length(stable_units)
                push!(stable_units, neuron)
            else  # neuron is linearly dependent

                S = collect(stable_units)
                alpha = transpose(W[layer][S, :]) \ W[layer][neuron, :]

                for neuron_next in 1:neuron_count[layer+1] # adjust weights and biases
                    W[layer+1][neuron_next, S] .+= W[layer+1][neuron_next, neuron] * alpha
                    b[layer+1][neuron_next] += W[layer+1][neuron_next, neuron] * (b[layer][neuron] - dot(b[layer][S], alpha))
                end

                push!(removed_neurons[layer], neuron)
            end
        else
            unstable_units = true
        end

    end

    if unstable_units == false # all units in the layer are stable
        println("Fully stable layer")

        if length(stable_units) > 0

            W_bar = Matrix{eltype(W[1][1])}(undef, neuron_count[layer+1], neuron_count[layer-1-layers_removed])
            b_bar = Vector{eltype(b[1][1])}(undef, neuron_count[layer+1])

            S = collect(stable_units)

            for neuron_next in 1:neuron_count[layer+1]

                b_bar[neuron_next] = b[layer+1][neuron_next] + dot(W[layer+1][neuron_next, S], b[layer][S])

                for neuron_previous in 1:neuron_count[layer-1-layers_removed]
                    W_bar[neuron_next, neuron_previous] = dot(W[layer+1][neuron_next, S], W[layer][S, neuron_previous])
                end
            end

            W[layer+1] = W_bar
            b[layer+1] = b_bar

            layers_removed += 1
            removed_neurons[layer] = 1:neuron_count[layer]
        else
            output = model((init_ub + init_lb) ./ 2)
            error("WHOLE NETWORK IS CONSTANT WITH OUTPUT: $output")
        end
    end

    println("Removed $(length(removed_neurons[layer]))/$(neuron_count[layer]) neurons")
    return layers_removed
end

"""
    function build_model!(W, b, K, neurons)

Builds a new `Flux.Chain` model from the given weights and biases.
Modifies the `W` and `b` arrays.

Returns the new `Flux.Chain` model.
"""
function build_model!(W, b, K, neurons)

    new_layers = [];
    layers = findall(neurons -> length(neurons) > 0, [neurons(l) for l in 1:K]) # layers with neurons
    for (i, layer) in enumerate(layers)

        W[layer] = W[layer][neurons(layer), neurons(i == 1 ? 0 : layers[i-1])]
        b[layer] = b[layer][neurons(layer)]

        if layer != last(layers)
            push!(new_layers, Dense(W[layer], b[layer], relu))
        else
            push!(new_layers, Dense(W[layer], b[layer]))
        end
    end

    return Flux.Chain(new_layers...)

end