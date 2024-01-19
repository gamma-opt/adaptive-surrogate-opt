using Flux
using Surrogates

# get the current MIP model solution x^*

# use generate_data to generate new samples around x^* and evaluate the function value at these samples

# retrain the surrogate model using the new data, considering freezing some layers parameters

# use the surrogate model to find the next x^*

# compute the function value at x^* and check its precision with the true function value (if within a certain threshold, then stop)

# return the best x^* found so far