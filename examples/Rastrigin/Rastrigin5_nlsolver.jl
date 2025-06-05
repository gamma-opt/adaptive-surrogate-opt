using Optim

# Define the negative Rastrigin function (for maximization)
function neg_rastrigin(x)
    n = length(x)
    return -(10n + sum(x.^2 - 10 .* cos.(2π .* x)))  # Negative for maximization
end

# Setup
n_dims = 5
x0 = fill(5.0, n_dims)  # Start from positive bound
lower_bounds = fill(-5.12, n_dims)
upper_bounds = fill(5.12, n_dims)

# Method 1: Particle Swarm
println("Using Particle Swarm Optimization:")
result_pso = optimize(neg_rastrigin, lower_bounds, upper_bounds,
                     ParticleSwarm(lower_bounds, upper_bounds, 100),
                     Optim.Options(show_trace = true,
                                 iterations = 10000,
                                 time_limit = 30.0))

println("\nPSO Results:")
println("x = ", Optim.minimizer(result_pso))
println("f(x) = ", -Optim.minimum(result_pso))  # Convert back to positive
println("Iterations: ", Optim.iterations(result_pso))

# Method 2: Simulated Annealing
println("\nUsing Simulated Annealing:")
result_sa = optimize(neg_rastrigin, lower_bounds, upper_bounds, x0,
                    SAMIN(),
                    Optim.Options(show_trace = true,
                                iterations = 10_000,
                                time_limit = 30.0))

println("\nSA Results:")
println("x = ", Optim.minimizer(result_sa))
println("f(x) = ", -Optim.minimum(result_sa))  # Convert back to positive
println("Iterations: ", Optim.iterations(result_sa))