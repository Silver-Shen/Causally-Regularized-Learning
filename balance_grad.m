function g_w = balance_grad(W, X)
    n = size(X, 1); % sample number
    m = size(X, 2); % feature number
    g_w = zeros(n, m);
    for i=1:m
        X_sub = X;
        X_sub(:,i) = 0; % the ith column is treatment
        I = double(X(:,i)>0);
        J1 = (X_sub'*((W.*W).*I))/((W.*W)'*I)...
            -(X_sub'*((W.*W).*(1-I)))/((W.*W)'*(1-I));
        dJ1W = 2*(X_sub'.*((W.*I)*ones(1,m))'*((W.*W)'*I)...
                  -X_sub'*((W.*W).*I)*(W.*I)')/((W.*W)'*I)^2 ...
              -2*(X_sub'.*((W.*(1-I))*ones(1,m))'*((W.*W)'*(1-I))...
                  -X_sub'*((W.*W).*(1-I))*(W.*(1-I))')/((W.*W)'*(1-I))^2;
        g_w(:,i) = 2*dJ1W'*J1;
    end
end