% generate sample points within the given upper and lower bounds using
% computeMaxTemp.m
% bounds = [Float64[120, 900, 20, 40, 30, 10], Float64[180, 1200, 40, 60, 50, 30]]
% Define the tuple x with 6 elements (pairs of upper and lower bounds)
% Define the tuple x with 6 elements (pairs of upper and lower bounds)
% Define the tuple x with 6 elements (pairs of upper and lower bounds)
% Define the bounds for each dimension
% Define the bounds for each dimension

function [sobol_samples, max_temp] = dataGeneration(bounds, num_samples)
    

    % Generate Sobol sequence
    sobol_points = sobolset(6, 'Skip', 1e3, 'Leap', 1e2);
    sobol_samples = net(sobol_points, num_samples);

    % Scale Sobol samples to match parameter bounds
    for i = 1:6
        sobol_samples(:, i) = sobol_samples(:, i) * (bounds(i, 2) - bounds(i, 1)) + bounds(i, 1);
    end

    % Initialize max_temp array to store the results
    max_temp = zeros(num_samples, 1);

    % Evaluate the function for each Sobol sample
    for i = 1:num_samples
        % Extract Sobol sample for each input parameter
        Tair_sample = sobol_samples(i, 1);
        Tgas_sample = sobol_samples(i, 2);
        hair_sample = sobol_samples(i, 3);
        hgaspressureside_sample = sobol_samples(i, 4);
        hgassuctionside_sample = sobol_samples(i, 5);
        hgastip_sample = sobol_samples(i, 6);

        % Compute max temperature using the provided function
        max_temp(i) = computeMaxTemp(Tair_sample, Tgas_sample, hair_sample, ...
            hgaspressureside_sample, hgassuctionside_sample, hgastip_sample);
        
        fprintf('Element %d : %d\n', i, max_temp(i));
       
    end
end



