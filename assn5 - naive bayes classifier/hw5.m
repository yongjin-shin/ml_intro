%% hw5 [logistic regression]
clear
bow = load('bow_new.mat');
bow = struct2array(bow);

% read csv file
train_csv = 'train-processed.csv';
test_csv = 'test-processed.csv';

var_name = {'label', 'word'};
tw_train = readtable(train_csv, 'Delimiter', ',');
tw_test = readtable(test_csv, 'Delimiter', ',');

tw_train.Properties.VariableNames = var_name;
tw_test.Properties.VariableNames = var_name; 

%% main
D = [4000]; %65538
n_D=size(D);num_D=n_D(2);
result_test = zeros(num_D, 1); result_train = zeros(num_D, 1);

for i=1:num_D
    str_D = string(D);
    [result_test(i,1), result_train(i,1)] = load_data(tw_train, tw_test, D(i), str_D(i), bow);
end

figure
plot(D, result_train, 'red')
hold on
plot(D, result_test, 'black')
% hline = refline([0 0.74]);


%% load data
function [result_test, result_train] = load_data(tw_train, tw_test, D, str_D, bow)
    
    train_file = strcat('train_top', str_D, '.mat');
    test_file = strcat('test_top', str_D, '.mat'); 

    % set label
    train_label = tw_train.label;
    test_label = tw_test.label;

    % load word2vec
    if exist(train_file, 'file') == 2
        train_data = load(train_file); train_data = struct2array(train_data);
        test_data = load(test_file); test_data = struct2array(test_data);
        train_data = train_data';
        test_data=test_data';
    else
        train_data = word2vecc(tw_train.word, bow, D); save(train_file, 'train_data');
        test_data = word2vecc(tw_test.word, bow, D); save(test_file, 'test_data');
        train_data = train_data';
        test_data=test_data';
    end
    
    % compute \theta and \pi
    [theta, pi] = findpara(train_data, train_label);
    
    % test the model
    nn_train = size(train_label);
    nn_test = size(test_label);
    
    y_train = testmodel(theta, pi, train_data);
    comp_train = y_train == train_label;
    result_train = sum(comp_train,1)/nn_train(1)
    
    y_test = testmodel(theta, pi, test_data);
    comp_test = y_test == test_label;
    result_test = sum(comp_test,1)/nn_test(1)
    
    return
end



%%
function [result] = testmodel(theta, pi, test_data)
    K = 2; % 0, 1
    N = size(test_data);
    D = N(1); % D: num of features
    N = N(2); % N: num of elements
    pi = pi'; test_data = test_data';
    L = ones(N,K); L = L .* log(pi);
    theta_ = 1-theta;
    
    tmp = test_data * log(theta);
    tmp_ = ~test_data * log(theta_);
    
    L = L + tmp + tmp_; 
    log_sum_result = logsum(L);
    
    L = L - log_sum_result;
    L = exp(L);
    result = L(:, 1) < L(:, 2);
    
    
    return
end

%% log-sum trick
function result = logsum(a)
    a_max = max(a, [], 2);
    a = a - a_max;
    tmp = exp(a);
    s = sum(tmp, 2);
    result = a_max + log(s);
    return
end

%% compute \theta and \pi
function [theta, pi] = findpara(word2vector, class_labels)
    % bow: bag of words
    K = 2; % 0, 1
    N = size(word2vector);
    D = N(1); % D: num of features
    N = N(2); % N: num of elements
    Nclass = zeros(K, 1); Nfeature = zeros(D, K);
    y = class_labels; x = word2vector;
    
    Nclass(2) = sum(y); Nclass(1) = N - Nclass(2);
    for k=1:2
        k = k-1;idx = y==k;
        k_class = x(:, idx);
        Nfeature(:, k+1) = sum(k_class, 2);
    end
    
    Nfeature(Nfeature == 0) = 1;
    pi = Nclass./N;
    theta = bsxfun(@rdivide, Nfeature, Nclass');
    return
end

%% make word2vec
function x = word2vecc(input_cell, bow, D) %tw_train.word
    bow = bow(1:D);
    n = size(input_cell); n = n(1);
    x = zeros(n,D);

    str = string(input_cell); str = lower(str);
    for i=1:n
        [~, ~, tmp_idx] = intersect(regexp(str(i), ' ', 'split'), bow);
        x(i, tmp_idx) = 1;
    end
    
    return
end

%% make bag of words w/top K
function bow = bagofwords(input, K)
    t = input.word;
    t = regexp(t, ' ', 'split');
    tt = horzcat(t{:});
    tt = string(tt);
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