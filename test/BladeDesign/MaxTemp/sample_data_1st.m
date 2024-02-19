% Number of sample points
num_samples = 375;

% Bounds for each parameter
bounds = [120, 142.5;    % Tair
          900, 1012.5;   % Tgas
          32.5, 40;   % hair
          40, 47.5;      % hgaspressureside
          30, 37.5;      % hgassuctionside
          10, 17.5];    % hgastip

[sobol_samples, max_temp] = dataGeneration(bounds, num_samples);

% Concatenate sobol_samples and max_temp into a single matrix
combined_data = [sobol_samples, max_temp];

% Write the combined data to a CSV file
writematrix(combined_data, 'combined_data_1st.csv')