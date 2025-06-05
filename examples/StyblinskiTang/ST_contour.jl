using GLMakie

function styblinski_tang_2d(x1, x2)
    return 0.5 * (x1^4 - 16x1^2 + 5x1 + x2^4 - 16x2^2 + 5x2)
end

# Create a grid of points
x1 = range(-5, 5, length=1000)
x2 = range(-5, 5, length=1000)

# Calculate z values
z = [styblinski_tang_2d(i, j) for i in x1, j in x2]

# Create the figure
fig = Figure(resolution=(800, 800))  # Make it square

# Create contour plot
ax = Axis(fig[1, 1], 
    xlabel="x₁", 
    ylabel="x₂",
    title="Styblinski-Tang Function Contour",
    aspect=1,  # Force aspect ratio to be 1:1
    limits=(-5, 5, -5, 5))  # Set exact limits for x and y axes

# Create contour plot
GLMakie.contour!(ax, x1, x2, z, 
    levels=100,  # number of contour levels
    colormap=:viridis)

# Add a colorbar
Colorbar(fig[1, 2], limits=(minimum(z), maximum(z)), 
    label="f(x₁, x₂)")

# Show the global minima points
global_min = -2.903534
GLMakie.scatter!(ax, [global_min], [global_min], 
    color=:red, 
    markersize=15,
    label="Global minimum")

axislegend(ax)

display(fig)