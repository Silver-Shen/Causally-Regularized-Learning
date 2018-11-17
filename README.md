# Causally-Regularized-Learning
This repo contains the core code for method described in ***Zheyan Shen, Peng Cui, Kun Kuang, Bo Li, and Peixuan Chen. 2018. Causally Regularized Learning with Agnostic Data Selection Bias. In 2018 ACM Mul- timedia Conference (MM ’18)***



Here is a simple demo showing how it works in Matlab:

```matlab
% Generate predictor X and outcome Y (binary)
X = 2*round(rand(1000, 20))-1; % 1000 samples and 20 features
beta_true = ones(20, 1);
Y = double(sigmoid(X*beta_true)>=0.5);
lambda0 = 1; %Logistic loss
lambda1 = 0.1; %Balancing loss
lambda2 = 1; %L_2 norm of sample weight
lambda3 = 0; %L_2 norm of beta
lambda4 = 0.001; %L_1 norm of bata
lambda5 = 1; %Normalization of sample weight
MAXITER = 1000;
ABSTOL = 1e-3;
W_init = rand(1000, 1);
beta_init = 0.5*ones(20, 1);
[W, beta, J_loss] = mainFunc(X, Y, ...
        lambda0, lambda1, lambda2, lambda3, lambda4, lambda5,...
        MAXITER, ABSTOL, W_init, beta_init);
```

