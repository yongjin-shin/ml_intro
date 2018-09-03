%% HW06 mog
clear
K = 7;
%%
plane = 'plane.jpg';
tiger = 'tiger.jpg';
boy = 'kid.jpg';

my_mog(tiger, K);


%%
function [] = my_mog(filename, K)
    area = solver(filename, K);
    for k = 1:K
        fig = imagesc(area(:,:,k))
        colorbar
        new_filename = strcat(erase(filename,'.jpg'), '_mog', string(K), '-', string(k), '.jpg'); 
        saveas(fig, new_filename);
    end
end

%%
function [area] = solver(filename, K)
    iter = 0;
    max_iter = 100000;
    img = imread(filename);
    img = rgb2lab(img);
    img_size = size(img); row = img_size(1); col = img_size(2);tot = row*col;
    sqz_img = squeeze(reshape(img, [1, tot, 3]));

    mean = my_kmeans(filename, img, K)
    pi = ones(K,1); pi=pi*1/K;
    resp = get_resp(K, pi, sqz_img, tot, mean);
    
    while(iter<max_iter)
        old_resp = resp;
        [mean, sigma, pi] = calculate(K, resp, sqz_img);
        resp = get_resp(K, pi, sqz_img, tot, mean, sigma);
        
        if abs(old_resp - resp) < 10^(-10)
            area = reshape(resp, [row, col, K]);
            mean
            sigma
            pi
            break;
        else
            iter = iter+1;
            continue;
        end
    end    
end

%%
function [mean, sigma, pi] = calculate(K, resp, sqz_img)
    %pi
    pi_sum = sum(resp, 1);
    tot_pi = sum(pi_sum, 2);
    pi = pi_sum/tot_pi; pi=pi';
    
    %mean
    mean = zeros(K,3);
    for k=1:K
        for i = 1:3
            mean(k,i) = sum(sqz_img(:,i).*resp(:,k),1)/pi_sum(k);
        end
    end
    
    %sigma
    sigma = zeros(3,3,K);
    for k=1:K
        r = resp(:, k); r = repmat(r, 1, 3);
        dif = sqz_img - mean(k,:);
        r_dif = dif.*r;
        sigma(:,:,k) = (r_dif'*dif)/pi_sum(k);
    end
    
    return
end

%%
function resp = get_resp(K, pi, sqz_img, tot, mean, sigma)
    if nargin < 6
        sigma = eye(3, 3); sigma = sigma*3;
        sigma = repmat(sigma, 1, 1, K);
    end
    
    y = zeros(tot, K);
    
    for k=1:K
        y(:,k) = mvnpdf(sqz_img(:,:), mean(k,:), sigma(:,:,k));
    end
    
    sum = y*pi;
    pi_long = repmat(pi', tot, 1);
    y = pi_long.*y;
    resp = y./sum;
    return 
end

%%
function [final_init] = my_kmeans(filename, img, K)
    [area, final_init] = solver_kmeans(img, K); final_init
    fig = imshow(label2rgb(area));
    new_filename = strcat(erase(filename,'.jpg'), '_kmeans', string(K), '.jpg'); 
    saveas(fig, new_filename);
    return
end

%%
function [area, final_init] = solver_kmeans(img, K)
    mm = zeros(3,2);
    for i=1:3
        mm(i,1) = min(min(img(:,:,i)));
        mm(i,2) = max(max(img(:,:,i)));
    end
    diff = mm(:,2) - mm(:,1);

    rng('shuffle')
    init = rand(3,K);
    init = init.*diff+mm(:,1);init = init';
    img_size = size(img); row = img_size(1); col = img_size(2);

    iter = 0;
    max_iter = 1000;

    while(iter<max_iter)
        [flag, next_init] = update(img, init,K, row, col);
        if flag
            area = comp_dist(img, next_init, K, row, col);
            final_init = next_init;
            break;
        else
            init = next_init;
            iter = iter+1;
            continue;
        end
    end
end

%%
function [flag, next_init] = update(img, init, K, row, col)
    
    old_init = init;
    next_init = zeros(size(old_init));
    
    resp = comp_dist(img,init, K, row, col);
    
    for k = 1:K
        tmp = resp == k;
        num = sum(sum(tmp,1),2);
        
        for i = 1:3
            if num ~= 0
                next_init(k,i) = sum(sum(img(:,:,i).*tmp,1),2)/num;
            else
                next_init(k,:) = old_init(k,:);
            end
        end
    end
    
    if next_init == old_init
        flag = 1;
    else
        flag = 0;
    end
    
    return
end

%%
function resp = comp_dist(img, init, K, row, col)
    d = zeros(row, col, K);
    for k = 1:K
        d(:,:,k) = lab_dist(img,init(k,:));
    end
    D = cat(4, d(:,:,:));
    [~, resp] = min(D, [], 3);
    return
end

%%
function dist = lab_dist(img, init)
    cal = zeros(size(img));
    for i = 1:3
        cal(:,:,i) = img(:,:,i) - init(i);
    end
    tmp = cal.*cal;
    dist = sum(tmp,3);
    return
end