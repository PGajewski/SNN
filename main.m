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
best_networks = cell(20,1);
for j=1:100
    fprintf('Iteration: %d\n', j);
    for i=1:20
        neuron_number=i;
        [net]=train_net(trainX',trainY',neuron_number);

        % Check loop.
        train_result = sim(net, trainX);
        test_result = sim(net, testX);

        train_mse = immse(train_result, trainY);
        test_mse = immse(test_result, testY);
        
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
            best_networks{i} = net;
        end
    end
end

for i=1:20
    fprintf('Mean mse for %d neurons:\n learn: %f\n test: %f\n', i, sum_mses(1,i)/100,  sum_mses(2,i)/100);
    fprintf('Minimal mse for %d neurons:\n learn: %f\n test: %f\n', i, min_mses(1,i),  min_mses(2,i));
end

figure();
plot(1:1:20, min_mses(1,:), 1:1:20, min_mses(2,:));
title('Minimal mse in set of network');
xlabel('hidden neurons');
ylabel('mse');
legend('train set', 'test set');

figure();
plot(1:1:20, sum_mses(1,:), 1:1:20, sum_mses(2,:));
title('Sum of mse in set of network');
xlabel('hidden neurons');
ylabel('mse');
legend('train set', 'test set');

%% Count leverages for best networks.
u_all = zeros(1,20);
for n=1:20
   %Count Jocobian.
   Z = calc_jacobian(best_networks{n},trainX');
   
   result = sim(best_networks{n}, trainX);
   
   s = size(Z);
   N = s(1,1);
   q = s(1,2);
   if rank(Z) == q
   %SVD factorization.
        fprintf('Found for size: %d\n', n);
        [U,W,V] = svd(Z, 'econ');
        leverages = zeros(1,N);
        for k = 1 : N
           for i = 1 : q
              summary = 0;
              for j=1 : q
                  summary = summary + Z(k,j)*V(j,i);
              end
              leverages(1,k) = leverages(1,k) + (summary/W(i,i))^2; 
           end
        end
        
        %Count Ep.
        Ep = 0;
        for i = 1 : N
           Ep = Ep + ((trainY(i)-result(i))/(1-leverages(i)))^2;
        end
        Ep = sqrt(Ep/N);
        
        %Count u.
        u = 0;
        for i = 1 : N
           u = u + sqrt((N*leverages(i))/q); 
        end
        u = u/N;
        u_all(n) = u;
        %Print histogram.
        figure();
        histogram(leverages);
        title(sprintf('q/N=%f, n=%d, u=%f, Ep=%f', q/N, n, u, Ep));
        
   else
        fprintf('Bad rand of Z matrix for network: %d\n', n);
   end
end
figure();
plot(1:8, u_all(1:8));
xlabel('hidden neurons number');
ylabel('u');
%% Simulate 150 network for best hidden layer size.
best_size_1 = 4;
test_networks_1 = cell(50,1);
best_u_1 = zeros(50,1);
best_Ep_1 = zeros(50,1);

for n = 1 : 50
    test_networks_1{n} = train_net(trainX',trainY', best_size_1);
    result = sim(test_networks_1{n}, trainX);
   
    %Count Jocobian.
    Z = calc_jacobian(test_networks_1{n},trainX');
   
   s = size(Z);
   N = s(1,1);
   q = s(1,2);
   if rank(Z) == q
   %SVD factorization.
        fprintf('Found for network: %d\n', n);
        [U,W,V] = svd(Z, 'econ');
        leverages = zeros(1,N);
        for k = 1 : N
           for i = 1 : q
              summary = 0;
              for j=1 : q
                  summary = summary + Z(k,j)*V(j,i);
              end
              leverages(1,k) = leverages(1,k) + (summary/W(i,i))^2; 
           end
        end
        
        %Count Ep.
        Ep = 0;
        for i = 1 : N
           Ep = Ep + ((trainY(i)-result(i))/(1-leverages(i)))^2;
        end
        best_Ep_1(n,1) = sqrt(Ep/N);
        
        %Count u.
        u = 0;
        for i = 1 : N
           u = u + sqrt((N*leverages(i))/q); 
        end
        best_u_1(n,1) = u/N;
        
   else
        fprintf('Bad rand of Z matrix for network: %d\n', n);
   end  
end

best_size_2 = 5;
test_networks_2 = cell(50,1);
best_u_2 = zeros(50,1);
best_Ep_2 = zeros(50,1);

for n = 1 : 50
    test_networks_2{n} = train_net(trainX',trainY', best_size_2);
    result = sim(test_networks_2{n}, trainX);
    
    %Count Jocobian.
    Z = calc_jacobian(test_networks_2{n},trainX');
    
   s = size(Z);
   N = s(1,1);
   q = s(1,2);
   if rank(Z) == q
   %SVD factorization.
        fprintf('Found for network: %d\n', n);
        [U,W,V] = svd(Z, 'econ');
        leverages = zeros(1,N);
        for k = 1 : N
           for i = 1 : q
              summary = 0;
              for j=1 : q
                  summary = summary + Z(k,j)*V(j,i);
              end
              leverages(1,k) = leverages(1,k) + (summary/W(i,i))^2; 
           end
        end
        
        %Count Ep.
        Ep = 0;
        for i = 1 : N
           Ep = Ep + ((trainY(i)-result(i))/(1-leverages(i)))^2;
        end
        best_Ep_2(n,1) = sqrt(Ep/N);
        
        %Count u.
        u = 0;
        for i = 1 : N
           u = u + sqrt((N*leverages(i))/q); 
        end
        best_u_2(n,1) = u/N;
        
   else
        fprintf('Bad rand of Z matrix for network: %d\n', n);
   end  
end

best_size_3 = 6;
test_networks_3 = cell(50,1);
best_u_3 = zeros(50,1);
best_Ep_3 = zeros(50,1);

for n = 1 : 50
    test_networks_3{n} = train_net(trainX',trainY', best_size_3);
    result = sim(test_networks_3{n}, trainX);
    
        %Count Jocobian.
    Z = calc_jacobian(test_networks_3{n},trainX');
    
   s = size(Z);
   N = s(1,1);
   q = s(1,2);
   if rank(Z) == q
   %SVD factorization.
        fprintf('Found for network: %d\n', n);
        [U,W,V] = svd(Z, 'econ');
        leverages = zeros(1,N);
        for k = 1 : N
           for i = 1 : q
              summary = 0;
              for j=1 : q
                  summary = summary + Z(k,j)*V(j,i);
              end
              leverages(1,k) = leverages(1,k) + (summary/W(i,i))^2; 
           end
        end
        
        %Count Ep.
        Ep = 0;
        for i = 1 : N
           Ep = Ep + ((trainY(i)-result(i))/(1-leverages(i)))^2;
        end
        best_Ep_3(n,1) = sqrt(Ep/N);
        
        %Count u.
        u = 0;
        for i = 1 : N
           u = u + sqrt((N*leverages(i))/q); 
        end
        best_u_3(n,1) = u/N;
        
   else
        fprintf('Bad rand of Z matrix for network: %d\n', n);
   end  
end

figure();
scatter(best_Ep_1, best_u_1);
hold on;
scatter(best_Ep_2, best_u_2);
hold on;
scatter(best_Ep_3, best_u_3);
xlabel('Ep');
ylabel('u');
legend('4 nuerons','5 neurons', '6 neurons');
xlim([0.1 0.35]);
ylim([0.9 1]);
%% Count confidence interval
chosen_network = test_networks_2{44};
result = sim(chosen_network, trainX);

        %Count Jocobian.
    Z = calc_jacobian(chosen_network,trainX');
    
   s = size(Z);
   N = s(1,1);
   q = s(1,2);
   %SVD factorization.
    [U,W,V] = svd(Z, 'econ');
    leverages = zeros(1,N);
    for k = 1 : N
       for i = 1 : q
          summary = 0;
          for j=1 : q
              summary = summary + Z(k,j)*V(j,i);
          end
          leverages(1,k) = leverages(1,k) + (summary/W(i,i))^2; 
       end
    end

    %t_stud value N-q =~ 184, 1-alfa = 0.99
t_stud = 2.34613;
s = 0;
for k = 1 : N
    s = s + (trainY(i) - result(k))^2;
end
s = sqrt(s)/(N-q);

g_min = result;
g_pred_min = result;
g_measure_min = result;

g_max = result;
g_pred_max = result;
g_measure_max = result;

for i = 1 : N
   g_measure_min(i) =  result(i) - t_stud*s;
   g_min(i) = result(i) - t_stud*s *sqrt(leverages(1,i));
   g_pred_min(i) =  result(i)- t_stud*s*sqrt(leverages(1,i)/(1-leverages(1,i)));
   
   g_measure_max(i) =  result(i) + t_stud*s;
   g_max(i) = result(i) + t_stud*s*sqrt(leverages(1,i));
   g_pred_max(i) = result(i)+ t_stud*s*sqrt(leverages(1,i)/(1-leverages(1,i)));
end 

%Sort values.
A = [trainX', result', g_measure_min', g_min', g_pred_min', g_measure_max', g_max', g_pred_max'];
A = sortrows(A,1);

figure();
plot(A(:,1), A(:,3));
hold on;
plot(A(:,1), A(:,2));
hold on;
plot(A(:,1), A(:,6));
hold on;
xlabel('x');
ylabel('y');
title('Function approximation with confidence interval for measuring output')

figure();
plot(A(:,1), A(:,4));
hold on;
plot(A(:,1), A(:,2));
hold on;
plot(A(:,1), A(:,7));
hold on;
xlabel('x');
ylabel('y');
title('Function approximation with confidence interval for output value')

figure();
plot(A(:,1), A(:,5));
hold on;
plot(A(:,1), A(:,2));
hold on;
plot(A(:,1), A(:,8));
hold on;
xlabel('x');
ylabel('y');
title('Function approximation with confidence interval for predicting output')

%% Check chosen network.
result = sim(chosen_network, testX);
immse(result, testY)
figure();
histogram(leverages);
chosen_network.IW{1}'
chosen_network.b{1}
chosen_network.LW{2,1}'
chosen_network.b{2}

