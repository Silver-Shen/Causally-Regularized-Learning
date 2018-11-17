function [W, beta, J_loss] = mainFunc(X, Y, ...
    lambda0, lambda1, lambda2, lambda3, lambda4, lambda5,...
    MAXITER, ABSTOL, W_init, beta_init)

%% Initialization
n = size(X, 1); % Sample size
m = size(X, 2); % Feature dimension
W = W_init;
W_prev = W;
beta = beta_init;
beta_prev = beta;

parameter_iter = 0.5;
J_loss = ones(MAXITER, 1)*(-1);

lambda_W = 1;
lambda_beta = 1;

W_All = zeros(n, MAXITER);
beta_All = zeros(m, MAXITER);

%% Optimization with gradient descent
for iter = 1:MAXITER
    % Update beta
    y = beta;
    beta = beta + (iter/(iter+3))*(beta-beta_prev); % fast proximal gradient
    f_base = J_cost(W, beta, X, Y, lambda0, lambda1, lambda2, lambda3, lambda5);
    grad_beta = lambda0*(((sigmoid(X*beta)-Y).*(W.*W))'*X)'...               
               +2*lambda3*beta;
    
    while 1
        z = prox_l1(beta - lambda_beta*grad_beta, lambda_beta*lambda4);
        if J_cost(W, z, X, Y, lambda0, lambda1, lambda2, lambda3, lambda5)...
           <= f_base + grad_beta'*(z-beta) ...
           + (1/(2*lambda_beta))*sum((z-beta).^2)
            break;
        end
        lambda_beta = parameter_iter*lambda_beta;
    end
    beta_prev = y;
    beta = z;
    
    % Update W
    y = W;
    W = W+(iter/(iter+3))*(W-W_prev);    
    f_base = J_cost(W, beta, X, Y, lambda0, lambda1, lambda2, lambda3, lambda5);
    
    grad_W = 2*lambda0*(log(1+exp(X*beta))-Y.*(X*beta)).*W...
            +lambda1*balance_grad(W, X)*ones(m,1)...
            +4*lambda2*W.*W.*W...           
            +4*lambda5*(sum(W.*W)-1)*W;
        
    while 1
        z = prox_l1(W-lambda_W*grad_W, 0);
        if J_cost(z, beta, X, Y, lambda0, lambda1, lambda2, lambda3, lambda5)...
                <= f_base + grad_W'*(z-W) ...
                + (1/(2*lambda_W))*sum((z-W).^2)
            break;
        end
        lambda_W = parameter_iter*lambda_W;
    end
    W_prev = y;
    W = z;    
    
    W_All(:,iter) = W;
    beta_All(:,iter) = beta;
    
    J_loss(iter) = J_cost(W, beta, X, Y, ....
                          lambda0, lambda1, lambda2, lambda3, lambda5)...
                 + lambda4*sum(abs(beta));
             
    if iter > 1 && abs(J_loss(iter) - J_loss(iter-1)) < ABSTOL || iter == MAXITER
        break
    end   
end    
end
