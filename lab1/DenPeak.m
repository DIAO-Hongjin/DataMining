% 数据输入
data1 = load('Aggregation_cluster=7.txt');
data2 = load('flame_cluster=2.txt');
data3 = load('Jain_cluster=2.txt');
data4 = load('Pathbased_cluster=3.txt');
data5 = load('Spiral_cluster=3.txt');
data6 = load('Mfeat.mat');

% 原始数据
% data = data2;
% data = zscore(data6.data_mor);  % 读取data_mor特征并进行归一化
% data = zscore(data6.data_zer);  % 读取data_zer特征并进行归一化
% data = zscore(data6.data_pix);  % 读取data_pix特征并进行归一化
% data = zscore(data6.data_kar);  % 读取data_kar特征并进行归一化
% data = zscore(data6.data_fou);  % 读取data_mor特征并进行归一化
data = zscore(data6.data_fac);  % 读取data_fou特征并进行归一化

[n,m] = size(data);

% 选取截断距离
percent = 3;  % 设置每个样本邻域内样本数量的平均值占总样本数的比例
DistanceMaztrix = pdist2(data,data);  % 距离矩阵
all_distance = [];
for i = 1:n   % 记录所有距离值
    all_distance = [all_distance DistanceMaztrix(i,(i+1):n)]; 
end
[distance_order, ~] = sort(all_distance);  % 对距离值进行排序
radius = distance_order(round(percent/100*n^2/2));  %按照比例选择截断距离

%radius = 1.3086;  
% 计算局部密度
% Cut-off kernel：邻域内的样本点数量
% rou = zeros(1,n);
% for i = 1:n
%     for j = 1:n
%         if DistanceMaztrix(i,j) <= radius
%             rou(i) = rou(i) + 1;
%         end
%     end
% end
% 计算局部密度
% Gaussian kernel
rou = zeros(1,n);
for i = 1:n
    for j = 1:n
        if i ~= j
            rou(i) = rou(i) + exp(-(DistanceMaztrix(i,j)/radius)^2);
        end
    end
end

% 计算相对距离
delta = zeros(1,n);
neighbor = zeros(1,n);
for i = 1:n
    [distance_order, index_order] = sort(DistanceMaztrix(i,:)); % 排序
    for j = 1:n
        if rou(index_order(j)) > rou(i)  % 找到距离最近的局部密度更高的样本点
            delta(i) = distance_order(j);
            neighbor(i) = index_order(j);
            break;
        end
    end
    if delta(i) == 0  %若样本点为全局密度最高，则相对距离为最大距离
        delta(i) = distance_order(n);
    end
end

% 画出决策图
figure(10000); 
plot(rou,delta,'.');
title ('Decision Graph')
xlabel ('\rho')
ylabel ('\delta')
rect = getrect(10000);  % 用鼠标从决策图中选取簇心
rou_min = rect(1);
delta_min = rect(2);
close all;

% 确定簇心
K = 0;                % 记录簇的数量
center_index = [];    % 记录簇心
centers = zeros(K,m);
for i = 1:n  % 局部密度和相对距离均大于鼠标选取的阈值时，该点被选择为簇心
    if rou(i) > rou_min && delta(i) > delta_min
        K = K+1;
        center_index = [center_index i];
    end
end

% 聚类过程
label = zeros(1,n);                  % 记录样本所属的类别
[~,rou_index] = sort(rou,'descend'); % 局部密度从高到低排序
for i = 1:K      % 先确定簇心所属的类别
    label(center_index(i)) = i;
end
for i = 1:n      % 样本的类别为与其最近的密度更高的点的类别
    if label(rou_index(i)) == 0
        label(rou_index(i)) = label(neighbor(rou_index(i)));
    end
end

% figure;  
% hold on;  
% for i = 1:n 
%     for j = 1:K
%         if label(i) == j
%             if j == 1
%                 plot(data(i,1),data(i,2),'.','Color',[1 0 0]);
%             elseif j == 2
%                 plot(data(i,1),data(i,2),'.','Color',[0 1 0]);
%             elseif j == 3
%                 plot(data(i,1),data(i,2),'.','Color',[0 0 1]);
%             elseif j == 4
%                 plot(data(i,1),data(i,2),'.','Color',[1 0 1]);
%             elseif j == 5
%                 plot(data(i,1),data(i,2),'.','Color',[1 1 0]);
%             elseif j == 6
%                 plot(data(i,1),data(i,2),'.','Color',[0 0 0]);
%             elseif j == 7
%                 plot(data(i,1),data(i,2),'.','Color',[0 1 1]);
%             elseif j == 8
%                 plot(data(i,1),data(i,2),'.','Color',[0 0.5 0.5]);
%             elseif j == 9
%                 plot(data(i,1),data(i,2),'.','Color',[0.5 0.5 0]);
%             elseif j == 10
%                 plot(data(i,1),data(i,2),'.','Color',[0.5 0 0.5]);
%             end
%         end
%         plot(data(center_index(j),1),data(center_index(j),2),'ko');
%     end
% end  
% grid on;  

result = ClusteringMeasure(label,data6.classid)