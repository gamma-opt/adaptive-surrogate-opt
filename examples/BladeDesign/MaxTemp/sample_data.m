% Number of sample points
num_samples = 1000;

% Bounds for each parameter
bounds = [120, 180;    % Tair
          900, 1200;   % Tgas
          20, 40;   % hair
          40, 60;      % hgaspressureside
          30, 50;      % hgassuctionside
          10, 30];    % hgastip

[sobol_samples, max_temp] = dataGeneration(bounds, num_samples);

% Concatenate sobol_samples and max_temp into a single matrix
combined_data = [sobol_samples, max_temp];

% Write the combined data to a CSV file
writematrix(combined_data, 'combined_data.csv')