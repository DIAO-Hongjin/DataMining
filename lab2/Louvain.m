cornell = load('dataset\cornell\cornell.mat');
texas = load('dataset\texas\texas.mat');
washington = load('dataset\washington\washington.mat');
wisconsin = load('dataset\wisconsin\wisconsin.mat');

% data = cornell;
% data = texas ;
% data = washington;
data = wisconsin;
[nVertex,nFeature] = size(data.F);

% 得到相似度矩阵
A = data.A;                                         % 使用邻接矩阵构造相似度矩阵
% A = 10./(pdist2(data.F,data.F,'euclidean')+10);     % 使用欧拉距离构造相似度矩阵（全连接）
% A = 200./(pdist2(data.F,data.F,'cityblock')+200);   % 使用街区距离构造相似度矩阵（全连接）
% A = 1-pdist2(data.F,data.F,'hamming');              % 使用汉明距离构造相似度矩阵（全连接）
% A = 1-pdist2(data.F,data.F,'jaccard');              % 使用Jaccard相似度构造相似度矩阵（全连接）
% A = 1-pdist2(data.F,data.F,'cosine');               % 使用余弦相似度构造相似度矩阵（全连接）
% A = A.*data.A;                                      % 给原有的边根据相似度赋值
% 取top_k相似度构造K近邻图
% top_k = 10;                                           
% [max_similarity, index_order] = sort(A,2,'descend');  % 对相似度进行排序
% A = zeros(nVertex);
% for i = 1:nVertex                                     % 构造K近邻图作为相似度矩阵
%     for j = 2:(top_k+1)
%         A(i,index_order(i,j)) = max_similarity(i,j);
%         A(index_order(i,j),i) = max_similarity(i,j);
%     end
% end
figure;
plot(graph(A));

% 合并结点
cur_label1 = 1:nVertex; % 每个结点初始化为不同的社区
for t = 1:10000
    new_label1 = cur_label1;
    for i = 1:nVertex              % 遍历节点
        neighbor = find(A(i,:));
        max_gain = -10000;
        max_index = 0;
        for j = 1:numel(neighbor)  % 尝试将结点i加入到其邻居所在的社区中
            temp_label = new_label1;
            temp_label(i) = temp_label(neighbor(j));
            gain = cal_modularity_gain(A,temp_label,i);
            if(gain > max_gain)
                max_gain = gain;
                max_index = neighbor(j);
            end
        end
        if max_gain > 0            %选择能使增益最大的邻居进行合并
            new_label1(i) = new_label1(max_index);
        end
    end
    if new_label1 == cur_label1    % 结点的社区归属不再改变时停止这一步骤
        break;
    else
        cur_label1 = new_label1;
    end
end

% 对图进行压缩
label_type = unique(cur_label1);
nCommunity = numel(label_type);
% 记录社区中的结点
community_vertex = zeros(nCommunity,nVertex);  
for i = 1:nCommunity
    actual_vertex = find(cur_label1==label_type(i));
    for j = 1:numel(actual_vertex)
        community_vertex(i,j) = actual_vertex(j);
    end
end
% 构造新的网络
compression = zeros(nCommunity,nCommunity);    
for i = 1:nCommunity
    community1 = community_vertex(i,find(community_vertex(i,:)));
    % 社区内结点之间的边的权重转化为新结点的环的权重
    compression(i,i) = (sum(sum(A(community1,community1)))...
        +sum(diag(A(community1,community1))))/2;
    % 社区间的边的权重转化为新结点之间的边的权重
    for j = (i+1):nCommunity
        community2 = community_vertex(j,find(community_vertex(j,:)));
        compression(i,j) = sum(sum(A(community1,community2)));
        compression(j,i) = sum(sum(A(community1,community2)));
    end
end
figure;
plot(graph(compression));

% 继续合并压缩后的图
cur_label2 = label_type;           % 社区的标签
final_label = cur_label1;          % 结点的标签
for t = 1:10000
    new_label2 = cur_label2;
    for i = 1:nCommunity           % 遍历社区
        neighbor = find(compression(i,:));
        max_gain = -10000;
        max_index = 0;
        for j = 1:numel(neighbor)  % 尝试合并两个相邻的社区
            temp_label = new_label2;
            temp_label(i) = temp_label(neighbor(j));
            gain = cal_modularity_gain(compression,temp_label,i);
            if(gain > max_gain)
                max_gain = gain;
                max_index = neighbor(j);
            end
        end
        if max_gain > 0            %选择能使增益最大的邻居进行合并
            new_label2(i) = new_label2(max_index);
            final_label(community_vertex(i,find(community_vertex(i,:))))...
                = new_label2(max_index);
        end
    end
    if new_label2 == cur_label2    % 结点的社区归属不再改变时停止这一步骤
        break;
    else
        cur_label2 = new_label2;
    end
end

% 画出最终结果
final_graph = zeros(nVertex);
for i = 1:nVertex
    for j = (i+1):nVertex
        if final_label(i) == final_label(j) && A(i,j) ~= 0
            final_graph(i,j) = 1;
            final_graph(j,i) = 1;
        end
    end
end
figure;
plot(graph(final_graph));

% 评估聚类结果
result = ClusteringMeasure(final_label,data.label)

% 计算模块度增益
function gain = cal_modularity_gain(A,label,i)
vertex_in = find(label==label(i));            % 找出结点i所属社区C中的所有结点
k_i = sum(A(i,:));                            % 计算与结点i相连的所有边的和
k_i_in = sum(A(i,vertex_in));                 % 计算结点i和社区C内部结点的边的权重之和
sum_tot = sum(sum(A(vertex_in,:)))-...        % 社区C内全部结点的边的权重之和
    (sum(sum(A(vertex_in,vertex_in)))-sum(diag(A(vertex_in,vertex_in))))/2;
m = (sum(sum(A))+sum(diag(A)))/2;             % 计算整个网络的边的权重之和
gain = (k_i_in/(2*m))-(sum_tot*k_i/(2*m^2));  % 根据公式计算模块度增益
end