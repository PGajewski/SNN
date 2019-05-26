close all;
clear all;

%Read data from file.
fileID = fopen('A_test_2.txt','r');
formatSpec = '%f %f';
testData = textscan(fileID, formatSpec);
testX = testData(1);
testX = testX{:}';
testY = testData(2);
testY = testY{:}';
fclose(fileID);

fileID = fopen('A_train_2.txt','r');
trainData = textscan(fileID, formatSpec);
trainX = trainData(1);
trainX = trainX{:}';
trainY = trainData(2);
trainY = trainY{:}';
fclose(fileID);

% Neural network - preparing.

%Normalize data and centralize data.
data_max_x = max(trainX);
data_min_x = min(trainX);

trainX = (trainX - (data_max_x + data_min_x)/2)*(2/(data_max_x-data_min_x));

data_max_y = max(trainY);
data_min_y = min(trainY);

trainY = (trainY - (data_max_y + data_min_y)/2)*(2/(data_max_y-data_min_y));

%Part train data


testX = (testX - (data_max_x + data_min_x)/2)*(2/(data_max_x-data_min_x));
testY = (testY - (data_max_y + data_min_y)/2)*(2/(data_max_y-data_min_y));

figure(1)
scatter(testX, testY, 2);
title('Test data after normalization');
xlabel('x');
ylabel('y');

figure(2)
scatter(trainX', trainY', 2);
title('Train data after normalization');
xlabel('x');
ylabel('y');
%% Find number of neurons.

sum_mses = zeros(2,20);
min_mses = ones(2,20);

for j=1:100
    fprintf('Iteration: %d\n', j);
    for i=1:20
        neuron_number=i;
        [net]=train_net(trainX',trainY',neuron_number);

        % Check loop.
        result = sim(net, trainX);
        train_difference = (result - trainY).^2;

        result = sim(net, testX);
        test_difference = (result - testY).^2;

        train_mse = mean(train_difference);
        test_mse = mean(test_difference);
        fprintf('Mean error for %d neurons:\n learn: %f\n test: %f\n', i, train_mse,  test_mse);
        sum_mses(1,i) = sum_mses(1,i) + train_mse;
        sum_mses(2,i) = sum_mses(2,i) + test_mse;
        
        %Save minimal value.
        if min_mses(1,i) > train_mse
            min_mses(1,i) = train_mse;
        end
        
        %Save minimal value.
        if min_mses(2,i) > test_mse
            min_mses(2,i) = test_mse;
        end
    end
end

for i=1:20
    fprintf('Mean mse for %d neurons:\n learn: %f\n test: %f\n', i, sum_mses(1,i)/100,  sum_mses(2,i)/100);
    fprintf('Minimal mse for %d neurons:\n learn: %f\n test: %f\n', i, min_mses(1,i),  min_mses(2,i));
end
