using JuMP, Flux
using Flux: params
using Gurobi

function JuMP_Model(NN::Chain, L_bounds::Vector{Float32}, U_bounds::Vector{Float32})

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

# get the current MIP model solution x^*

function get_x_star(MILP_model::Model)

    return [value.(MILP_model[:x][0,i]) for i in 1:length(MILP_model[:x][0,:])]

end