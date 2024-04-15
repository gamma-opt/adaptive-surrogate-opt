using JuMP, Flux
using Flux: params
using Gurobi

function JuMP_Model_raw(NN::Chain, L_bounds::Vector{Float32}, U_bounds::Vector{Float32})

    K = (length(NN) - 1) ÷ 2

    # store the weights and biases
    ps = params(NN)
    W = [ps[2*i-1] for i in 1:K]
    b = [ps[2*i] for i in 1:K]

    # stores the node count
    node_count = [if k == 1 length(ps[1][1, :]) else length(ps[2*(k-1)]) end for k in 1:K+1]
    
    final_L_bounds = copy(L_bounds)
    final_U_bounds = copy(U_bounds)

    # final_L_bounds, final_U_bounds = bound_tightening(NN, U_bounds, L_bounds, bt_verbose)

    # create a JuMP model
    model = Model(optimizer_with_attributes(Gurobi.Optimizer))

    # sets the variables
    @variable(model, x[k in 0:K, j in 1:node_count[k+1]] >= 0)
    if K > 1 # s and z variables only to hidden layers, i.e., layers 1:K-1
        @variable(model, s[k in 1:K, j in 1:node_count[k+1]] >= 0)
        @variable(model, z[k in 1:K, j in 1:node_count[k+1]], Bin)
    end
    @variable(model, U[k in 0:K, j in 1:node_count[k+1]])
    @variable(model, L[k in 0:K, j in 1:node_count[k+1]])

    # fix values to all U[k,j] and L[k,j] from U_bounds and L_bounds
    index = 1
    for k in 0:K
        for j in 1:node_count[k+1]
            fix(U[k, j], final_U_bounds[index])
            fix(L[k, j], final_L_bounds[index])
            index += 1
        end
    end   

    # constraints corresponding to the ReLU activation functions
    for k in 1:K
        for node in 1:node_count[k+1] # node count of the next layer of k, i.e., the layer k+1
            temp_sum = sum(W[k][node, j] * x[k-1, j] for j in 1:node_count[k])
            if k < K # hidden layers: k = 1, ..., K-1
                @constraint(model, temp_sum + b[k][node] == x[k, node] - s[k, node])
            else # output layer: k == K
                @constraint(model, temp_sum + b[k][node] == x[k, node])
            end
        end
    end

    # fix bounds U and L to input layer nodes
    for input_node in 1:node_count[1]
        delete_lower_bound(x[0, input_node])
        @constraint(model, L[0, input_node] <= x[0, input_node])
        @constraint(model, x[0, input_node] <= U[0, input_node])
    end

    # fix bounds to the hidden layer nodes
    @constraint(model, [k in 1:K, j in 1:node_count[k+1]], x[k, j] <= U[k, j] * z[k, j])
    @constraint(model, [k in 1:K, j in 1:node_count[k+1]], s[k, j] <= -L[k, j] * (1 - z[k, j]))

    # fix bounds to the output layer nodes
    for output_node in 1:node_count[K+1]
        delete_lower_bound(x[K, output_node])
        @constraint(model, L[K, output_node] <= x[K, output_node])
        @constraint(model, x[K, output_node] <= U[K, output_node])
    end

    @objective(model, Min, x[K, node_count[K+1]]) # arbitrary objective function to have a complete JuMP model

    return model

end

function JuMP_Model(NN::Chain, L_bounds::Vector{Float32}, U_bounds::Vector{Float32})

    K = (length(NN) - 1) ÷ 2    # dropouts are not considered as layers

    # store the weights and biases
    ps = params(NN)
    W = [ps[2*i-1] for i in 1:K]
    b = [ps[2*i] for i in 1:K]

    # stores the node count
    node_count = [if k == 1 length(ps[1][1, :]) else length(ps[2*(k-1)]) end for k in 1:K+1]
    
    final_L_bounds = copy(L_bounds)
    final_U_bounds = copy(U_bounds)

    # final_L_bounds, final_U_bounds = bound_tightening(NN, U_bounds, L_bounds, bt_verbose)

    # create a JuMP model
    model = Model(optimizer_with_attributes(Gurobi.Optimizer))

    # sets the variables
    @variable(model, x[k in 0:K, j in 1:node_count[k+1]] >= 0)
    if K > 1 # s and z variables only to hidden layers, i.e., layers 1:K-1
        @variable(model, s[k in 1:K, j in 1:node_count[k+1]] >= 0)
        @variable(model, z[k in 1:K, j in 1:node_count[k+1]], Bin)
    end
    @variable(model, U[k in 0:K, j in 1:node_count[k+1]])
    @variable(model, L[k in 0:K, j in 1:node_count[k+1]])

    # fix values to all U[k,j] and L[k,j] from U_bounds and L_bounds
    index = 1
    for k in 0:K
        for j in 1:node_count[k+1]
            fix(U[k, j], final_U_bounds[index])
            fix(L[k, j], final_L_bounds[index])
            index += 1
        end
    end 

    # constraints corresponding to the ReLU activation functions
    temp_sum = Dict()
    for k in 1:K  # Assuming node_count is a list or array
        temp_sum[k] = Dict()
        for node in 1:node_count[k+1]
            # Calculate the sum for each combination of k and node
            temp_sum[k][node] = sum(W[k][node, j] * x[k-1, j] for j in 1:node_count[k])
        end
    end
    @constraint(model, con_hidden[k = 1:K-1, node = 1:node_count[k+1]], 
                temp_sum[k][node] + b[k][node] == x[k, node] - s[k, node])  # hidden layer constraints
    
    @constraint(model, con_output[k = K, node = 1:node_count[k+1]], 
                temp_sum[K][node] + b[K][node] == x[K, node])               # output layer constraints


    # fix bounds U and L to input layer nodes
    foreach(delete_lower_bound, x[0, 1:node_count[1]])
    @constraint(model, lb_input[input_node = 1:node_count[1]], x[0, input_node] >= L[0, input_node])
    @constraint(model, ub_input[input_node = 1:node_count[1]], x[0, input_node] <= U[0, input_node])

    # fix bounds to the hidden layer nodes
    @constraint(model, [k in 1:K, j in 1:node_count[k+1]], x[k, j] <= U[k, j] * z[k, j])
    @constraint(model, [k in 1:K, j in 1:node_count[k+1]], s[k, j] <= -L[k, j] * (1 - z[k, j]))

    # fix bounds to the output layer nodes
    for output_node in 1:node_count[K+1]
        delete_lower_bound(x[K, output_node])   # delete the lower bound of the output node, default is 0
        @constraint(model, L[K, output_node] <= x[K, output_node])
        @constraint(model, x[K, output_node] <= U[K, output_node])
    end

    @objective(model, Min, x[K, node_count[K+1]]) # arbitrary objective function to have a complete JuMP model

    return model

end

# reuse the old MILP model and rebuild it with the new weights and biases of the unfrozen layers

function rebuild_JuMP_Model(retrained_model::Chain, MILP_model::Model, freeze::Int, L_bounds::Vector{Float32}, U_bounds::Vector{Float32})
	
    K = (length(retrained_model) - 1) ÷ 2    # dropouts are not considered as layers

    # store the weights and biases of the unfrozen layers
    ps = params(retrained_model)
    W = [ps[2*i-1] for i in freeze+1:K]     
    b = [ps[2*i] for i in freeze+1:K]       

    # stores the node count
    node_count = [if k == 1 length(ps[1][1, :]) else length(ps[2*(k-1)]) end for k in 1:K+1]
  
    # 1.fectch the bounds of the input layer nodes
    
    index = 1
    for j in 1:node_count[1]
        fix(MILP_model[:U][0, j], U_bounds[index])
        fix(MILP_model[:L][0, j], L_bounds[index])
        index += 1
    end
    
    # 2. change the bounds of the input layers

    JuMP.delete(MILP_model, MILP_model[:lb_input])  
    JuMP.delete(MILP_model, MILP_model[:ub_input])
    unregister(MILP_model, :lb_input)   # call unregister to remove the symbolic reference after calling delete
    unregister(MILP_model, :ub_input)
    @constraint(MILP_model, lb_input[input_node = 1:node_count[1]], MILP_model[:x][0, input_node] >= MILP_model[:L][0, input_node])
    @constraint(MILP_model, ub_input[input_node = 1:node_count[1]], MILP_model[:x][0, input_node] <= MILP_model[:U][0, input_node])


    # 3. change the constraints of MILP_model corresponding to the unfrozen layers' weights and biases

    for k in freeze+1:K-1, node in 1:node_count[k+1]
        JuMP.delete(MILP_model, MILP_model[:con_hidden][k, node])
    end
    for node in 1:node_count[K+1]
        JuMP.delete(MILP_model, MILP_model[:con_output][K, node])
    end
    unregister(MILP_model, :con_hidden)
    unregister(MILP_model, :con_output)

    # constraints corresponding to the ReLU activation functions
    temp_sum = Dict()
    for k in freeze+1:K  # Assuming node_count is a list or array
        temp_sum[k] = Dict()
        for node in 1:node_count[k+1]
            # Calculate the sum for each combination of k and node
            temp_sum[k][node] = sum(W[k-freeze][node, j] * MILP_model[:x][k-1, j] for j in 1:node_count[k])
        end
    end

    @constraint(MILP_model, con_hidden[k = freeze+1:K-1, node = 1:node_count[k+1]],     # hidden layer constraints
                temp_sum[k][node] + b[k-freeze][node] == MILP_model[:x][k, node] - MILP_model[:s][k, node])  
    
    @constraint(MILP_model, con_output[k = K, node = 1:node_count[k+1]],                # output layer constraints
                temp_sum[K][node] + b[K-freeze][node] == MILP_model[:x][K, node])               

    return MILP_model
    
end

# set start values of the variables corresponding to the input layers using the solution of the previous MILP model
function warmstart_JuMP_Model(MILP_model::Model, x_star::Vector{Float64})

    for i in 1:length(x_star)
        set_start_value(MILP_model[:x][0, i], x_star[i])
    end

    return MILP_model

end

# store multiple solutions in the solution pool
function sol_pool(MILP_model::Model, num_solutions::Int; mean = 0, std = 1)

    solution_pool_x = []
    solution_pool_f = []

    for i in 1:num_solutions
        println("Solution $i:")
        println("   x = ", value.(MILP_model[:x][0,i] for i in 1:length(MILP_model[:x][0,:]); result = i) .* std .+ mean)
        push!(solution_pool_x, value.(MILP_model[:x][0,i] for i in 1:length(MILP_model[:x][0,:]); result = i))  # store the normalised solution
        println(" obj = ", objective_value(MILP_model; result = i))
        push!(solution_pool_f, objective_value(MILP_model; result = i))
    end

    return solution_pool_x, solution_pool_f

end