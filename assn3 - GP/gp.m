load("gp_data.mat"); train = x;
min = min(train(:,1)); max = max(train(:,1))+1; n = 10^(2);
test = linspace(min, max, n); test = test';
hp1 = [1, 1, 0.1];
hp2 = [0.3, 1.08, 0.00005];
hp3 = [3.0, 1.16, 0.89];
hp4 = [3, 1.16, 0.0009];
[mean, var] = solver(hp4, train, test);


%%
f = [mean+sqrt(var(1,:)'); flipdim(mean-sqrt(var(1,:)'),1)];
fill([test; flipdim(test,1)], f, [7 7 7]/8)
hold on; plot(test, mean); plot(train(:,1), train(:,2), '+')


%%
function [mean, var] = solver(hp, train, test)
    K = compute_kernel(hp(1), hp(2), hp(3), train(:,1), train(:,1));
    k_ = compute_kernel(hp(1), hp(2), hp(3), test(:,1), train(:,1));
    L = chol(K,'lower');
    alpha = L'\(L\train(:,2));
    mean = k_ * alpha;
    v = L\k_';
    var = RBF(hp(2), hp(3), test(:,1), test(:,1)) - dot(v,v);
    
    return
end
%%
function K = compute_kernel(len, sig, prec, X1, X2)
    len1 = size(X1,1);
    len2 = size(X2,1);
    K = zeros(len1, len2);
    I = eye(len1, len2);
    
    for i = 1:len1
        K(i,:) = RBF(sig, len, X1(i), X2);    
    end
    
    if(len1 == len2)
        K = K + prec^2*I;
    end
        
    return
end

%%
function k = RBF(sig, len, x1, X2)
    diff = x1 - X2;
    in = diff.*diff;
    k = sig^2.*exp(-1*in/(len^2));
    return
end