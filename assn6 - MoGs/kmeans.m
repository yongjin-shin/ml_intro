%% HW06 Kmeans
clear
K = 5;

%%
plane = 'plane.jpg';
tiger = 'tiger.jpg';

my_kmeans(plane, K);

%%
function [] = my_kmeans(filename, K)
    area = solver(filename, K);
    fig = image(area);
    new_filename = strcat(erase(filename,'.jpg'), '_kmeans.jpg'); 
    saveas(fig, new_filename);
end

%%
function [area, final_init] = solver(filename, K)
    img = imread(filename);
    img = rgb2lab(img);

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