function f_x = balance_cost(W, X)
    m = size(X, 2); % feature number
    f_x = zeros(m,1);
    for i=1:m
        X_sub = X;
        X_sub(:,i) = 0; % the ith column is treatment
        I = double(X(:,i)>0);
        loss = (X_sub'*((W.*W).*I))/((W.*W)'*I)...
              -(X_sub'*((W.*W).*(1-I)))/((W.*W)'*(1-I));       
        f_x(i) = loss'*loss;
    end    
end