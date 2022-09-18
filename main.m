clc
clear all
close all
format long

%% Question 1, 2, 3

X = load('iris.dat'); %load data
class = X(:,end);
X = X(:,1:size(X,2)-1);
X = X./10;

m = 1.7; % fuzzifier value
jj = 3; % Number of clusters

% First Initialization % All Points at origin
for k = 1:100
    V = zeros(jj,size(X,2));
    [U,V,iter] = myFCM(X,jj,m,V);
    acc_origin(k,1) = accuracy(U,class);
end

% Second Initialization % Multivariate gaussian 
for k = 1:100
    mu = zeros(jj,size(X,2));
    sigma = ones(1,size(X,2));
    V = mvnrnd(mu,sigma);
    [U,V,iter] = myFCM(X,jj,m,V);
    acc_mvgaussian(k,1) = accuracy(U,class);
end

% Third Initialization % Random vectors from the data
for k = 1:100
    init = randperm(size(X,1));
    V = X(init(1:jj),:);
    [U,V,iter] = myFCM(X,jj,m,V);
    acc_datapoints(k,1) = accuracy(U,class);
end

Initialization_Method = {'All points at origin';...
    'Multivariate Gaussian (Zero Mean and Unit Variance)';...
    'Random points from the data'};
Mean_Accuracy = [mean(acc_origin);mean(acc_mvgaussian);mean(acc_datapoints)];
Std_Accuracy = [std(acc_origin);std(acc_mvgaussian);std(acc_datapoints)];
disp('Accuracy using FCM with various Initializations')
T1 = table(Initialization_Method,Mean_Accuracy,Std_Accuracy)

%% Question 4
% From table we can see that Multivariable Gaussian works best for initilization

fuzzifier = [1.1; 1.3; 1.5; 1.7; 2; 2.5; 5; 10; 100];
for i = 1:max(size(fuzzifier))
    m = fuzzifier(i);
    for k = 1:100
        % Multivariate gaussian Initialization
        mu = zeros(jj,size(X,2));
        sigma = ones(1,size(X,2));
        V = mvnrnd(mu,sigma);
        [U,V,iter] = myFCM(X,jj,m,V);
        acc_mvgaussian_m(k,i) = accuracy(U,class);
    end
end

Fuzzifier_Value = fuzzifier;
Mean_Accuracy = zeros(max(size(fuzzifier)),1);
Std_Accuracy = zeros(max(size(fuzzifier)),1);
for i = 1:max(size(fuzzifier))
    Mean_Accuracy(i,1) = mean(acc_mvgaussian_m(:,i));
    Std_Accuracy(i,1) = std(acc_mvgaussian_m(:,i));
end
disp('Accuracy using FCM with Multivariate Gaussian Initialization and various values of Fuzzifier')
T2 = table(Fuzzifier_Value,Mean_Accuracy,Std_Accuracy)

%% Question 5
% GKFCM

X = load('iris.dat'); %load data
class = X(:,end);
X = X(:,1:size(X,2)-1);
X = X./10;

m = 1.7; % Fuzzifier Value
jj = 3; % Number of Clusters
ro = [0.1; 0.5; 1; 2; 5; 10; 100]; 
% Value of p
for jj = 1:max(size(ro))
    for tt = 1:100
        p = ro(jj,1) * ones(jj,1);
        mu = zeros(jj,size(X,2));
        sigma = ones(1,size(X,2));
        V = mvnrnd(mu,sigma);
        [U,V,S] = myGKFCM(X,jj,m,p,V);
        acc_gkfcm_p(tt,jj) = accuracy(U,class);
    end
end

P_Value = ro;
Mean_Accuracy = zeros(max(size(ro)),1);
Std_Accuracy = zeros(max(size(ro)),1);
for i = 1:max(size(ro))
    Mean_Accuracy(i,1) = mean(acc_gkfcm_p(:,i));
    Std_Accuracy(i,1) = std(acc_gkfcm_p(:,i));
end
disp('Accuracy using GKFCM with Multivariate Gaussian Initialization and various values of P')
T3 = table(P_Value,Mean_Accuracy,Std_Accuracy)

%% Question 6
% FCM algorithm with varying c

X = load('iris.dat'); %load data
class = X(:,end);
X = X(:,1:size(X,2)-1);
X = X./10;

m = 1.7; %fuzzifier value

% Varying Number of clusters
clusters = 2:20;
for jj = 1:max(size(clusters))
    c = clusters(jj);
    % Multivariate gaussian initialization
    for k = 1:100
        mu = zeros(c,size(X,2));
        sigma = ones(1,size(X,2));
        V = mvnrnd(mu,sigma);
        [U,V,iter] = myFCM(X,c,m,V);
        acc_varying_c(k,jj) = accuracy(U,class);
    end
end

mean_clusters = mean(acc_varying_c,1);
std_clusters = std(acc_varying_c,1);

errorbar(clusters,mean_clusters,std_clusters)
