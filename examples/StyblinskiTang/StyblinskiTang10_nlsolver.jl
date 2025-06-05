using Optim

# Define the Styblinski-Tang function
function styblinski_tang(x)
    return 0.5 * sum(x.^4 - 16x.^2 + 5x)
end

# Set up the optimization problem
n_dims = 5
x0 = fill(-5.0, n_dims)  # Starting point at corner
lower_bounds = fill(-5.0, n_dims)
upper_bounds = fill(5.0, n_dims)

# Run the optimization with Particle Swarm
# ParticleSwarm requires lower and upper bounds, and number of particles
result = optimize(styblinski_tang, lower_bounds, upper_bounds, 
                 ParticleSwarm(lower_bounds, upper_bounds, 50),  # 50 particles
                 Optim.Options(show_trace = true,
                             iterations = 1000,
                             time_limit = 30.0))

# Print results
println("\nOptimization Results:")
println("x = ", Optim.minimizer(result))
println("f(x) = ", Optim.minimum(result))
println("Converged: ", Optim.converged(result))
println("Number of iterations: ", Optim.iterations(result))