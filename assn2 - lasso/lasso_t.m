M = csvread("prostate.csv");
for col = 1:8
    mean_col = mean(M(:, col));
    M(:,col) = (M(:,col) - mean_col)/std(M(:,col));
end
M(:,9) = M(:,9) - mean(M(:,9));
Train = M(M(:, 10) == 1, :); Train(:, 10) = [];

lambda_min = 10^(-6);
lambda_max = 10^(4);
nlambda = 10^(4);
c = linspace(lambda_min, lambda_max, nlambda);
X = Train(:, 1:8);Y = Train(:, 9);
lstack = zeros(size(X,2), size(c, 2));
tt = zeros(size(X,2), size(c, 2));

n = 1;
for a = c
    lstack(:, n) = lasso_(X, Y, a);
    t = solveLasso(Y, X, a);
    tt(:, n) = t.beta;
    n = n+1;
end

x_axis = sum(abs(tt),1)/sum(abs((X'*X)\X'*Y));
plot(x_axis, tt)


%test = lasso(X, Y);
%plot(x_axis(1:100), test)



function w = lasso_(X, Y, lambda_)
    w = (X'*X + 2*lambda_)\(X'*Y);
    finished = 0;
    Tol = 10^(-6);
    D = size(X,2);

    while(finished == 0)
        w_old = w;
        for i = 1:D
            xi = X(:, 1);
            yi = (Y-X*w) + xi*w(i);
            deltai = (xi'*yi);
            
            if (deltai < -lambda_)
                w(i) = (deltai + lambda_)/(xi'*xi);
            elseif (deltai > lambda_)
                w(i) = (deltai - lambda_)/(xi'*xi);
            else
                w(i) = 0;
            end
        end
        
        if(max(abs(w-w_old)) <= Tol)
            finished = 1;
        end
        
    end
    return
end