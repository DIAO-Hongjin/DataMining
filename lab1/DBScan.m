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

[n,m] = size(data);

Eps = 4;
MinPts = 48;

% 密度阈值
% MinPts = 7;    
% % 邻域半径
% Eps = ((prod(max(data)-min(data))*MinPts*gamma(.5*m+1))/(n*sqrt(pi.^m))).^(1/m);

% 识别样本点的种类
% pattern为DBSCAN模型：n个样本点，m个维度
% 第m+1维代表样本点的种类：-1、0、1分别代表噪音点、边界点和核心点
% 第m+2维代表样本点所属的簇：-1、0、K分别代表噪音点、未搜索、第K类
pattern = zeros(n,m+2);         
pattern(:,1:m) = data(:,:);
Pts = zeros(1,n);                       % 样本点的密度
DistanceMaztrix = pdist2(data,data);    % 数据集的距离矩阵
for i = 1:n     % 所有样本先标记为噪音点
    pattern(i,m+1) = -1;
end
for i = 1:n     % 搜索每个样本点的邻域，记录样本点的密度
    % 记录邻域范围内的样本点个数
    neighbors = find(DistanceMaztrix(i,:)<=Eps);  
    Pts(i) = numel(neighbors);
    % 若密度大于密度阈值，则样本点种类更新为核心点
    % 且核心点的邻域范围内的非核心样本点的种类更新为边界点
    if Pts(i) >= MinPts
        pattern(i,m+1) = 1;             % 核心点
        for j = 1:Pts(i)
            if pattern(neighbors(j),m+1) ~= 1
                pattern(neighbors(j),m+1) = 0; % 边界点
            end
        end
    end
end

% figure;  
% hold on;  
% for i = 1:n
%     if pattern(i,m+1) == 1
%         plot(pattern(i,1),pattern(i,2),'.','Color',[1 0 0]);
%     elseif pattern(i,m+1) == 0
%         plot(pattern(i,1),pattern(i,2),'.','Color',[0 1 0]);
%     elseif pattern(i,m+1) == -1
%         plot(pattern(i,1),pattern(i,2),'.','Color',[0 0 1]);
%     else
%         plot(pattern(i,1),pattern(i,2),'.','Color',[0 0 0]);
%     end
% end

% 聚类过程
K = 0;     % 记录簇的数量
for i = 1:n     % 遍历样本点
    if pattern(i,m+2) == 0               % 处理未搜索过的样本点
        if pattern(i,m+1) == 1           % 未搜索过的核心点
            K = K+1;                     % 簇的数量加一
            pattern(i,m+2) = K;          % 这个未搜索过的核心点归属为新的簇
            neighbors = find(DistanceMaztrix(i,:)<=Eps);     % 搜索邻域
            cnt = 1;
            while true
                j = neighbors(cnt);
                % 邻域内的点归属于簇K
                % 若邻域内存在未搜索过的核心点，则继续搜索这个核心点的邻域
                if pattern(j,m+2) == 0
                    pattern(j,m+2) = K;
                    if pattern(j,m+1) == 1     
                        neighbors = [neighbors find(DistanceMaztrix(j,:)<=Eps)];
                    end
                end
                cnt = cnt+1;
                if cnt > numel(neighbors) % 邻域搜索完毕
                    break;
                end
            end
        elseif pattern(i,m+1) == -1      % 样本点为噪音点，则所属类别也为噪音点
            pattern(i,m+2) = -1;
        end
    end
end

% figure;  
% hold on;  
% for i = 1:n 
%     for j = -1:K
%         if pattern(i,m+2) == j
%             if j == -1
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0 0 0]);
%             elseif j == 1
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
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0 1 1]);
%             elseif j == 7
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0.5 0.5 0.5]);
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

result = ClusteringMeasure(pattern(:,m+2),data6.classid)