using Surrogates
using Flux
using Random
using Statistics
using JuMP
using BSON
using Plots

include("../src/NNSurrogate.jl")
include("../src/NNJuMP.jl")

"""
- Styblinski-Tang function (2 variables)
"""

# set random seed
Random.seed!(1)

# the function we are going to build surrogate for
styblinski_tang(x::Tuple) = 0.5 * sum([xi^4 - 16*xi^2 + 5*xi for xi in x])

data_ST = generate_data(styblinski_tang, [fill(-4.0, 2), fill(-2.0, 2)], 1000, SobolSample(), 0.8)
data_ST_scale, μ, σ = normalise_data(data_ST)   # normalise the data, as μ ≈ 0 and σ ≈ 1, the normalisation is not necessary

config1_ST = NN_Config([2,512,256,1], [relu, relu, identity], false, 0.1, 0.1, Flux.Optimise.Optimiser(Adam(0.1, (0.9, 0.999)), ExpDecay(1.0)), 1, 800, 5000, 0)
result_ST = NN_train(data_ST, config1_ST)
NN_results(config1_ST, result_ST)

model_init = result_ST.model
BSON.@save "ST2_model_init.bson" model_init
BSON.@load "ST2_model_init.bson" model_init

L_bounds = vcat(fill(Float32(-4.0), 2), fill(Float32(-1e6), 769))
U_bounds = vcat(fill(Float32(-2.0), 2), fill(Float32(1e6), 769))

MILP_model = JuMP_Model(model_init, L_bounds, U_bounds)
optimize!(MILP_model)
f_hat_init = objective_value(MILP_model)
x_star_init = get_x_star(MILP_model)
f_init = styblinski_tang(Tuple(x_star_init))

# Theoretical optimal solution
x_opt = [-2.90354,-2.90354]
f_opt = styblinski_tang(Tuple(x_opt))

# visualise the surrogate model
x, y = Float32.(-4.0:0.002:-2.0), Float32.(-4.0:0.002:-2.0)
z_true = [styblinski_tang((x1,x2)) for x1 in x, x2 in y]
z_surrogate = [model_init([x1, x2])[1] for x1 in x, x2 in y]

p1 = surface(x, y, z_true, clims = (-80, -20), colorbar = false)
scatter!(p1, [x_opt[1]], [x_opt[2]], [f_opt], label = "")
p2 = surface(x, y, z_surrogate, clims = (-80, -20), colorbar = false)
scatter!(p2, [x_star_init[1]], [x_star_init[2]], [f_hat_init], label = "")
plot(p1, p2, 
    title=["True function" "Surrogate model"], 
    zlims = (-80, -20),
    legend = false
)
savefig("ST2_surrogate_surface.png")

p1 = contour(x, y, z_true, clims = (-80, -20))
scatter!(p1, [x_opt[1]], [x_opt[2]], label = "")
p2 = contour(x, y, z_surrogate, clims = (-80, -20))
scatter!(p2, [x_star_init[1]], [x_star_init[2]], label = "")
plot(p1, p2, 
    title=["True function" "Surrogate model"], 
    legend = false
)
savefig("ST2_surrogate_contour.png")