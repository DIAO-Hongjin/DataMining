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
pattern = zeros(n,m+1);
pattern(:,1:m) = data(:,:);

% 簇心初始化
% 原始方法：随机选择K个样本点作为簇心
% centers = zeros(K,m);
% for i = 1:K   
%     centers(i,:) = data(randi(n),:);  
% end

% 簇心初始化
% K-means++：与现有簇心的最小距离越大的样本点越有可能被选择为簇心
% centers = zeros(K,m);
% centers(1,:) = data(randi(n),:);    % 随机选择第一个簇心
% for i = 2:K    %选择剩余的簇心
%     temp_distance = zeros(n,i-1);   % 记录样本点与当前簇心的距离
%     temp_min = zeros(1,n);          % 记录样本点与当前簇心的最小距离 
%     distance_sum = 0;               % 记录样本点与当前簇心的最小距离之和
%     for j = 1:n     % 计算样本点与当前簇心的距离
%         for k = 1:i-1
%             temp_distance(j,k) = norm(data(j,:)-centers(k,:));
%         end
%     end
%     for j = 1:n     % 计算样本点与当前簇心的最小距离
%         temp_min(j) =  min(temp_distance(j,:));
%         distance_sum = distance_sum+temp_min(j);
%     end
%     % 使用轮盘赌算法，以距离为概率选择作为下一个簇心的样本点
%     random_num = distance_sum*rand; % 相当于轮盘的指针
%     for j = 1:n     % 根据随机数的大小选择对应样本点作为下一个簇心
%         random_num = random_num - temp_min(j);
%         if random_num <= 0     % 指针所指的样本点
%             centers(i,:) = data(j,:);
%             break;
%         end
%     end
% end

% 簇心初始化
% 与现有簇心的最小距离最大的样本点被选择为簇心
centers = zeros(K,m);
centers(1,:) = data(randi(n),:);    % 随机选择第一个簇心
for i = 2:K    %选择剩余的簇心
    temp_distance = zeros(n,i-1);   % 记录样本点与当前簇心的距离
    temp_min = zeros(1,n);          % 记录样本点与当前簇心的最小距离
    for j = 1:n     % 计算样本点与当前簇心的距离
        for k = 1:i-1
            temp_distance(j,k) = norm(data(j,:)-centers(k,:));
        end
    end
    for j = 1:n     % 计算样本点与当前簇心的最小距离
        temp_min(j) =  min(temp_distance(j,:));
    end
    % 与当前簇心的最小距离最大的样本点被选择为下一个簇心
    [temp_max, index] = max(temp_min);
    centers(i,:) = data(index,:);
end

 % plot(data(:,1),data(:,2),'+',centers(:,1),centers(:,2),'ko');

% 目标函数的优化过程
for t = 1: 10000    % 最大迭代次数为10000次
    distance = zeros(n,K);      % 记录每个样本点与每个质心的距离
    num = zeros(1,K);           % 记录每个簇中样本点的数量
    new_centers = zeros(K,m);   % 记录新的簇心

    for i = 1:n     % 计算样本点与每个簇心的距离
        for j = 1:K
            distance(i,j) = norm(data(i,:)-centers(j,:));
        end
    end
    for i = 1:n     % 将样本点划分到最近的簇心所属的类别中
        [min_distance,index] =  min(distance(i,:));
        pattern(i,m+1) = index;
    end
    
    for i = 1:K     % 重新计算簇心
        for j = 1:n
            if pattern(j,m+1) == i
                new_centers(i,:) = new_centers(i,:)+pattern(j,1:m);
                num(i) = num(i)+1;
            end
        end
        new_centers(i,:) = new_centers(i,:)/num(i);
    end
    
    if new_centers == centers  % 簇心不再变化，说明收敛，退出循环
        break;
    else                       % 仍未收敛，更新簇心，继续迭代 
        centers = new_centers;
    end
end

% 人工数据集，画图评估聚类结果
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
%             plot(centers(j,1),centers(j,2),'ko');
%         end
%     end
% end  
% grid on;  

% 真实数据集 评测聚类结果
result = ClusteringMeasure(pattern(:,m+1),data6.classid)
