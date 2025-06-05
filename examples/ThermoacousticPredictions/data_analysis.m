% Load the dataset from a CSV file
filename = './test/ThermoacousticPredictions/data_init.csv';
% Load the dataset
data = csvread('data_init.csv'); % Adjust the row and column indices if necessary

% Extract the 6th column
column6 = data(:, 6);

% Plot the histogram
histogram(column6);

% Label your axes
xlabel('Value');
ylabel('Frequency');
title('Histogram of Growth Rate Data');

% Calculate the mean
meanValue = mean(column6);

% Calculate the standard deviation
stdValue = std(column6);

% Display the results
fprintf('Mean: %f\n', meanValue);
fprintf('Standard deviation: %f\n', stdValue);