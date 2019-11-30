%数据输入
data1 = load('Aggregation_cluster=7.txt');
data2 = load('flame_cluster=2.txt');
data3 = load('Jain_cluster=2.txt');
data4 = load('Pathbased_cluster=3.txt');
data5 = load('Spiral_cluster=3.txt');
data6 = load('Mfeat.mat');

data = data4;

[n,m] = size(data);

max_silhouette = -1;
final_K = 0;

% for K = 2:10
%     [pattren,centers,silhouette] = KMeansFunction(K,data);
%     [K,silhouette]
%     if silhouette > max_silhouette
%         final_K = K;
%         final_pattren = pattren;
%         final_centers = centers;
%         max_silhouette = silhouette;
%     end
% end

% 根据轮廓系数确定K值
for K = 2:10     % 遍历K从2到10时的情况
    avg_silhouette = 0;     % 记录每个K值下的平均轮廓系数
    for i = 1:30     % 对于每个K值，重复30次以避免陷入局部最优解
        [pattren,centers,silhouette] = KMeansFunction(K,data);
        avg_silhouette = avg_silhouette+silhouette;
    end
    avg_silhouette = avg_silhouette/30;    % 计算平均轮廓系数
    if avg_silhouette > max_silhouette     % 记录平均轮廓系数最大的K值
        final_K = K;
        max_silhouette = avg_silhouette;
    end
end

% 以平均轮廓系数最大的K值作为簇的数量，进行最后一次聚类，得到最终结果
[final_pattren,final_centers,silhouette] = KMeansFunction(final_K,data);

figure;  
hold on;  
for i = 1:n 
    for j = 1:final_K
        if final_pattren(i,m+1) == j
            if j == 1
                plot(final_pattren(i,1),final_pattren(i,2),'*','Color',[1 0 0]);
            elseif j == 2
                plot(final_pattren(i,1),final_pattren(i,2),'*','Color',[0 1 0]);
            elseif j == 3
                plot(final_pattren(i,1),final_pattren(i,2),'*','Color',[0 0 1]);
            elseif j == 4
                plot(final_pattren(i,1),final_pattren(i,2),'*','Color',[1 0 1]);
            elseif j == 5
                plot(final_pattren(i,1),final_pattren(i,2),'*','Color',[1 1 0]);
            elseif j == 6
                plot(final_pattren(i,1),final_pattren(i,2),'*','Color',[0 0 0]);
            elseif j == 7
                plot(final_pattren(i,1),final_pattren(i,2),'*','Color',[0 1 1]);
            elseif j == 8
                plot(final_pattren(i,1),final_pattren(i,2),'*','Color',[0 0.5 0.5]);
            elseif j == 9
                plot(final_pattren(i,1),final_pattren(i,2),'*','Color',[0.5 0.5 0]);
            elseif j == 10
                plot(final_pattren(i,1),final_pattren(i,2),'*','Color',[0.5 0 0.5]);
            end
            plot(final_centers(j,1),final_centers(j,2),'ko');
        end
    end
end  
grid on;  
