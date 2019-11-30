% 数据输入
data1 = load('Aggregation_cluster=7.txt');
data2 = load('flame_cluster=2.txt');
data3 = load('Jain_cluster=2.txt');
data4 = load('Pathbased_cluster=3.txt');
data5 = load('Spiral_cluster=3.txt');
data6 = load('Mfeat.mat');

% 原始数据
% data = data5;
% data = zscore(data6.data_mor);  % 读取data_mor特征并进行归一化
% data = zscore(data6.data_zer);  % 读取data_zer特征并进行归一化
% data = zscore(data6.data_pix);  % 读取data_pix特征并进行归一化
% data = zscore(data6.data_kar);  % 读取data_kar特征并进行归一化
% data = zscore(data6.data_fou);  % 读取data_mor特征并进行归一化
data = zscore(data6.data_fac);  % 读取data_fou特征并进行归一化

K = 10;
[n,m] = size(data);

% 使用高斯相似度构造相似度矩阵
sigma = 10;
W = zeros(n,n);
for i = 1:n  
    for j = 1:n
        W(i,j) = exp(-(norm(data(i,:)-data(j,:))^2)/(2*(sigma)^2));
    end
end

D = zeros(n,n);
for i = 1:n
    D(i,i) = sum(W(i,:));
end

% RCut
% L = D-W;                      % 构造非标准化拉普拉斯矩阵
% [U, ~] = eigs(L, K, 'SM');    % 求解拉普拉斯矩阵的前K小特征向量

% NCut
L = eye(n)-(D^(-1/2) * W * D^(-1/2));  % 构造标准化拉普拉斯矩阵
[U, ~] = eigs(L, K, 'SM');             % 求解拉普拉斯矩阵的前K小特征向量
for i = 1:n                            % 将每个样本点的特征向量进行归一化
    U(i,:) = U(i,:)/norm(U(i,:));
end

pattern = zeros(n,m+1);
pattern(:,1:m) = data(:,:);
centers = zeros(K,K);
centers(1,:) = U(randi(n),:); 
for i = 2:K
    temp_distance = zeros(n,i-1);
    temp_min = zeros(1,n);
    for j = 1:n
        for k = 1:i-1
            temp_distance(j,k) = norm(U(j,:)-centers(k,:));
        end
    end
    for j = 1:n
        temp_min(j) =  min(temp_distance(j,:));
    end
    [temp_max, index] = max(temp_min);
    centers(i,:) = U(index,:);
end

for t = 1: 500000
    distance = zeros(n,K);
    num = zeros(1,K);
    new_centers = zeros(K,K);
    
    for i = 1:n
        for j = 1:K
            distance(i,j) = norm(U(i,:)-centers(j,:));
        end
    end
    for i = 1:n
        [min_distance,index] =  min(distance(i,:));
        pattern(i,m+1) = index;
    end
    
    for i = 1:K
        for j = 1:n
            if pattern(j,m+1) == i
                new_centers(i,:) = new_centers(i,:)+U(j,:);
                num(i) = num(i)+1;
            end
        end
        new_centers(i,:) = new_centers(i,:)/num(i);
    end
    
    if new_centers == centers
        % disp(['第',num2str(t),'次迭代时收敛'])
        break;
    else
        centers = new_centers;
    end
end

% figure;  
% hold on;  
% for i = 1:n 
%     for j = 1:K
%         if pattern(i,m+1) == j
%             if j == 1
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[1 0 0]);
%             elseif j == 2
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0 1 0]);
%             elseif j == 3
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0 0 1]);
%             elseif j == 4
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[1 0 1]);
%             elseif j == 5
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[1 1 0]);
%             elseif j == 6
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0 0 0]);
%             elseif j == 7
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0 1 1]);
%             elseif j == 8
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0 0.5 0.5]);
%             elseif j == 9
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0.5 0.5 0]);
%             elseif j == 10
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0.5 0 0.5]);
%             end
%         end
%     end
% end  
% grid on;  

result = ClusteringMeasure(pattern(:,m+1),data6.classid)