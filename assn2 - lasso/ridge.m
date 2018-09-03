M = csvread("prostate.csv");
for col = 1:8
    mean_col = mean(M(:, col));
    M(:,col) = (M(:,col) - mean_col)/std(M(:,col));
end
M(:,9) = M(:,9) - mean(M(:,9));

Test = M(M(:, 10) == 0, :); Test(:, 10) = [];
Train = M(M(:, 10) == 1, :); Train(:, 10) = [];

c = linspace(10^(-6), 10^(5), 10^(5));
y_size = size(Train, 2)-1;
x_size = size(c, 2);
rstack = zeros(y_size, x_size);
df = zeros(1, x_size);
n = 1;
for a = c
    rstack(:, n) = ridge_(Train(:, 1:8), Train(:, 9), a);
    df(n) = dof(Train(:, 1:8), a);
    n = n+1;
end

figure
hold on
plot(df,rstack)
title('Ridge Regression')
xlabel('Effective Dof')
ylabel('Coefficients')
legend('lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45')

function theta = ridge_(x, y, alpha)
    I = eye(size(x, 2));
    tmp = alpha*I;
    theta = (x'*x+tmp)\x'*y;
    return
end

function df = dof(x, alpha)
    I = eye(size(x, 2));
    tmp = alpha*I;
    result = (x'*x+tmp)\x';
    df = trace(x*result);
    return
end
    

    