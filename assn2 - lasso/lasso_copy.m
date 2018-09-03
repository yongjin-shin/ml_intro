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
    %t = solveLasso(Y, X, a);
    %tt(:, n) = t.beta;
    n = n+1;
end

x_axis = sum(abs(lstack),1)/sum(abs((X'*X)\X'*Y));
figure
plot(x_axis,lstack,'-o','MarkerIndices',1:4:length(Y))
title('Lasso Regression')
xlabel('Shrinkage Factor s')
ylabel('Coefficients')
legend('lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45')

function w = lasso_(X, Y, lambda_)
    D = size(X,2);
    w = (X'*X)\(X'*Y);
    finished = 0;
    Tol = 10^(-6);
    
    while(finished == 0)
        w_old = w;
         for d = 1:D
             x_d = X; w_d = w; 
             xd=X(:, d); x_d(:, d)=[]; w_d(d, :)=[];
             alpha = xd'*xd;
             beta = sum((Y - x_d*w_d)'*xd);
             
             if beta < -lambda_
                 w(d) = (beta+lambda_)/alpha;
             elseif beta > lambda_
                 w(d) = (beta-lambda_)/alpha;
             else
                 w(d) = 0;
             end
         end
        
        if(max(abs(w-w_old)) <= Tol)
            finished = 1;
        end
        
    end
    return
end