function [pattern,centers,silhouette] = KMeansFunction(K,data)
[n,m] = size(data);
pattern = zeros(n,m+1);
pattern(:,1:m) = data(:,:);

centers = zeros(K,m);
centers(1,:) = data(randi(n),:); 
for i = 2:K
    temp_distance = zeros(n,i-1);
    temp_min = zeros(1,n);
    for j = 1:n
        for k = 1:i-1
            temp_distance(j,k) = norm(data(j,:)-centers(k,:));
        end
    end
    for j = 1:n
        temp_min(j) =  min(temp_distance(j,:));
    end
    [~,index] = max(temp_min);
    centers(i,:) = data(index,:);
end

for t = 1: 500000
    distance = zeros(n,K);
    num = zeros(1,K);
    new_centers = zeros(K,m);
    
    for i = 1:n
        for j = 1:K
            distance(i,j) = norm(data(i,:)-centers(j,:));
        end
    end
    for i = 1:n
        [~,index] =  min(distance(i,:));
        pattern(i,m+1) = index;
    end
    
    for i = 1:K
        for j = 1:n
            if pattern(j,m+1) == i
                new_centers(i,:) = new_centers(i,:)+pattern(j,1:m);
                num(i) = num(i)+1;
            end
        end
        new_centers(i,:) = new_centers(i,:)/num(i);
    end
    
    if new_centers == centers
        break;
    else
        centers = new_centers;
    end
end

% 计算轮廓度
a = zeros(n,1);     % 簇内不相似度
b = zeros(n,K-1);   % 簇间不相似度
s = zeros(n,1);     % 样本点的轮廓系数
silhouette = 0;     % 聚类结果的轮廓系数
for i = 1:n
    same_num = 0;
    diff_num = zeros(1,K-1);
    % 遍历样本点，计算每个样本点的簇内不相似度和簇间不相似度
    for j = 1:n     
       if pattern(i,m+1) == pattern(j,m+1) 
           a(i) = a(i)+norm(data(i,:)-data(j,:));
           same_num = same_num + 1;
       else
           for k = 1:K
               if pattern(j,m+1) == k && k < pattern(i,m+1)
                   b(i,k) = b(i,k)+norm(data(i,:)-data(j,:));
                   diff_num(k) = diff_num(k)+1;
               elseif pattern(j,m+1) == k && k > pattern(i,m+1)
                   b(i,k-1) = b(i,k-1)+norm(data(i,:)-data(j,:));
                   diff_num(k-1) = diff_num(k-1)+1;
               end
           end
       end
    end
    % 簇内不相似度为样本点与簇内其他样本的平均距离
    a(i) = a(i)/same_num;
    % 簇间不相似度为样本点与不同簇中样本的平均距离的最小值
    for j = 1:K-1
        b(i,j) = b(i,j)/diff_num(j);
    end
    bi = min(b(i,:));
    % 计算样本的轮廓系数
    if a(i) > bi
        s(i) = (bi-a(i))/a(i);
    else
        s(i) = (bi-a(i))/bi;
    end
end
% 聚类结果的轮廓系数为样本的轮廓系数的平均值
for i = 1:n
    silhouette = silhouette+s(i);
end
silhouette = silhouette/n;
end

