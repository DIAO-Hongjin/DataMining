cornell = load('dataset\cornell\cornell.mat');
texas = load('dataset\texas\texas.mat');
washington = load('dataset\washington\washington.mat');
wisconsin = load('dataset\wisconsin\wisconsin.mat');
cora_udir = load('dataset\cora_udir.mat');
polblog = load('dataset\polblog.mat');
polbook = load('dataset\polbook.mat');

data = cornell;
% data = texas ;
% data = washington;
% data = wisconsin;
[nVertex,nFeature] = size(data.F);
% data = polbook;
% [nVertex,~] = size(data.A);

% 得到相似度矩阵
% W = data.A;                                         % 使用邻接矩阵构造相似度矩阵
% W = 10./(pdist2(data.F,data.F,'euclidean')+10);     % 使用欧拉距离构造相似度矩阵（全连接）
% W = 200./(pdist2(data.F,data.F,'cityblock')+200);   % 使用街区距离构造相似度矩阵（全连接）
% W = 1-pdist2(data.F,data.F,'hamming');              % 使用汉明距离构造相似度矩阵（全连接）
% W = 1-pdist2(data.F,data.F,'jaccard');              % 使用Jaccard相似度构造相似度矩阵（全连接）
W = 1-pdist2(data.F,data.F,'cosine');               % 使用余弦相似度构造相似度矩阵（全连接）
% 取top_k相似度构造K近邻图
top_k = 60;                                           
[max_similarity, index_order] = sort(W,2,'descend');  % 对相似度进行排序
W = zeros(nVertex);
for i = 1:nVertex                                     % 构造K近邻图作为相似度矩阵
    for j = 2:(top_k+1)
        W(i,index_order(i,j)) = max_similarity(i,j);
        W(index_order(i,j),i) = max_similarity(i,j);
    end
end
figure;
plot(graph(W));

% 计算标准化拉普拉斯矩阵
D = diag(sum(W));
L = D^-0.5*(D-W)*D^-0.5;

% 对特征向量按特征值从小到大排序
[V,D1] = eig(L);
[D_sort,index1] = sort(diag(D1));
D_sort = D_sort(index1);
V_sort = V(:,index1);

% 取前n小特征值对应的的特征向量
n = 20;
Y = V_sort(:,1:n);
for i = 1:nVertex
    Y(i,:) = Y(i,:)/norm(Y(i,:));
end

% 使用K-means进行预处理
k = 40;
% 选择相对较远的点作为簇心
cur_center = zeros(k,n);
cur_center(1,:) = Y(randi(nVertex),:);
for i = 2:k
    distance = pdist2(Y,cur_center(1:i-1,:));
    [~,index2] = max(min(distance,[],2));
    cur_center(i,:) = Y(index2,:);
end
% 优化K-means目标函数进行聚类
label = zeros(nVertex,1);
for t = 1:10000
    % 给每个点分配簇
    distance = pdist2(Y,cur_center);
    [~,index3] = min(distance,[],2);
    label = index3;
    % 重新计算簇心
    new_center = zeros(k,n);
    cluster_size = zeros(k,1);
    for i = 1:k
        for j = 1:nVertex
            if label(j) == i
                new_center(i,:) = new_center(i,:)+Y(j,:);
                cluster_size(i) = cluster_size(i)+1;
            end
        end
        new_center(i,:) = new_center(i,:)/cluster_size(i);
    end
    % 判断是否收敛
    if new_center == cur_center
        break;
    else
        cur_center = new_center;
    end
end
% % 画出K-means聚类后的结果
% kmeans_label = label;
% temp_graph = zeros(nVertex);
% for i = 1:nVertex
%     for j = (i+1):nVertex
%         if kmeans_label(i) == kmeans_label(j)
%             temp_graph(i,j) = 1;
%             temp_graph(j,i) = 1;
%         end
%     end
% end
% figure;
% plot(graph(temp_graph));

% 合并策略得到最终簇划分
nCommunity = 5;            % 指定最终划分的社区的数量
for t = 1:k-nCommunity     % 在社区数量达到指定个数前，合并社区
    ci = 0;
    cj = 0;
    min_ncut = 10000;
    label_type = unique(label);
    % 尝试将目前的社区进行两两合并
    for i = 1:numel(label_type)
        for j = (i+1):numel(label_type)
            temp_label = label;
            temp_label(find(temp_label==label_type(j))) = label_type(i);
            temp_ncut = cal_ncut(W,temp_label);    % 合并社区后重新计算NCut
            if(temp_ncut < min_ncut)    % 记录下使NCut最小的一次合并
                ci = label_type(i);
                cj = label_type(j);
                min_ncut = temp_ncut;
            end
        end
    end
    label(find(label==cj)) = ci;        % 执行使NCut最小的一次合并
end
% 画出最终结果
final_graph = zeros(nVertex);
for i = 1:nVertex
    for j = (i+1):nVertex
        if label(i) == label(j)
            final_graph(i,j) = 1;
            final_graph(j,i) = 1;
        end
    end
end
figure;
plot(graph(final_graph));

% 评估聚类结果
result = ClusteringMeasure(label,data.label)

% 计算NCut的值
function ncut = cal_ncut(W,label)
ncut = 0;
label_type = unique(label);
for i = 1:numel(label_type)
    A = find(label==label_type(i));  % 找到社区内的结点
    C_A = find(label~=label_type(i));% 找到社区外的结点
    cut = sum(sum(W(A,C_A)));        % 计算社区间的边的权重之和
    assoc = sum(sum(W(A,:)));        % 计算社区内结点与所有结点的边的权重之和
    ncut = ncut+cut/assoc;           % 计算NCut
end
end