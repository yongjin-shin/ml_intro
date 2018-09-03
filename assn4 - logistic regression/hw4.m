%% hw4 [logistic regression]
clear

%% grid search
K = [10]; 
threshold = [0.50];
n_K=size(K);num_K=n_K(2);
n_thres=size(threshold);num_thres = n_thres(2);
result_test = zeros(num_K, num_thres); result_train = zeros(num_K, num_thres);

for i=1:num_K
    for j = 1:num_thres
        str_K = string(K); str_thres = string(threshold);
        [result_test(i,j), result_train(i,j)] = gridsearch(K(i), threshold(j), str_K(i), str_thres(j));
    end
end

figure
plot(K, result_train, 'red')
hold on
plot(K, result_test, 'green')
hline = refline([0 0.74]);

function [result, result_train] = gridsearch(K, threshold, str_K, str_thres)
    
    train_csv = 'train-processed.csv';
    test_csv = 'test-processed.csv';

    var_name = {'label', 'word'};
    tw_train = readtable(train_csv, 'Delimiter', ',');
    tw_test = readtable(test_csv, 'Delimiter', ',');

    tw_train.Properties.VariableNames = var_name;
    tw_test.Properties.VariableNames = var_name; 

    train_file = strcat('train_top', str_K, '.mat'); test_file = strcat('test_top', str_K, '.mat'); 
    w_file = strcat('w_top', str_K, '.mat');
    compare_file = strcat('compare_top', str_K, '_', str_thres, '.mat');
    
    if exist(train_file, 'file') == 2
        train_data = load(train_file); train_data = struct2array(train_data);
        test_data = load(test_file); test_data = struct2array(test_data);
        train_data = train_data';
        test_data=test_data';
    else
        bow = bagofwords(tw_train, K);
        train_data = word2vecc(tw_train.word, bow, K); save(train_file, 'train_data');
        test_data = word2vecc(tw_test.word, bow, K); save(test_file, 'test_data');
        train_data = train_data';
        test_data=test_data';
    end

    train_label = tw_train.label;
    test_label = tw_test.label;

    if exist(w_file, 'file') == 2
        w = load(w_file); w = struct2array(w);
    else
        w = irls(K, train_data, train_label); save(w_file, 'w');
    end

    y_train = testmodel(w, train_data, threshold);
    comp_train = y_train == train_label;
    nn_train = size(train_label);
    result_train = sum(comp_train,1)/nn_train(1)
    
    y = testmodel(w, test_data, threshold);
    comp = y == test_label; save(compare_file, 'comp_train');
    nn = size(test_label);
    result = sum(comp,1)/nn(1)
    return
end

%%
function y = testmodel(w, x, threshold)
    z = x'*w;
    e = arrayfun(@sigmoid, z);
    y = e >= threshold;
    
    function e = sigmoid(t)
        e = 1/(1+exp((-1)*(t)));
        return
    end
end



%%
function w = irls(K, x, y)
    mx_iter = 10^5;
    n = size(x); n = n(2);
    w = zeros(K,1);
    w0 = log(mean(y)/(1-mean(y)));
    
    sigma = zeros(n,1);
    f = zeros(n,1);
    tmp1 = zeros(K,n);
%     tmp1 = zeros(K,K);
    
    for j = 1:mx_iter
        w_old = w;
        
        for i = 1:n
            f(i) = w'*x(:,i)+w0;
            sigma(i) = arrayfun(@sigmoid, f(i));
        end
    
        s = sigma.*(1-sigma);
        z = f + (y-sigma)./s;
                
%         tmp = x*x';
%         for k = 1:K
%              tmp1(:,k) = tmp(:,k)*s(k);
%         end
%         tmp2 = tmp1\x;
        
        for k = 1:n
            tmp1(:,k) = x(:,k)*s(k);
        end
        tmp3 = tmp1*x';
        
        reg = 10^(-5);
        test = reg*eye(K);
        tmp3 = tmp3+test;
        
        %cholesky
        L = chol(tmp3, 'lower');
        tmp2 = L'\(L\x);
        
        for k = 1:n
            tmp1(:,k) = tmp2(:,k)*s(k);
        end        
        
        w = tmp1*z;
        
        thres = ones(K,1); thres = thres*10^(-5);
        if abs(w - w_old) < thres
            break;
        end

    end
    return
    
    function e = sigmoid(t)
        e = 1/(1+exp((-1)*(t)));
        return
    end
end





%% make bag of words w/top K
function bow = bagofwords(input, K)
    tt = string(regexp([input.word{:}], ' ', 'split'));%tw_train.word{:}
    tt = tt'; tt=lower(tt);
    [words, ~, idx] = unique(tt);
    numOccurrences = histcounts(idx,numel(words));
    [~,rankIndex] = sort(numOccurrences,'descend');
    wordsByFrequency = words(rankIndex);

    st = stopWords;
    [common, ~, ~] = intersect(st, wordsByFrequency);
    bow = setxor(wordsByFrequency, common, 'stable');
    bow = bow(1:K);
end

%% make word2vec
function x = word2vecc(input_cell, bow, K) %tw_train.word
    n = size(input_cell); n = n(1);
    x = zeros(n,K);

    str = string(input_cell);
    for i=1:n
        [~, ~, tmp_idx] = intersect(regexp(str(i), ' ', 'split'), bow);
        x(i, tmp_idx) = 1;
    end
    
    return
end