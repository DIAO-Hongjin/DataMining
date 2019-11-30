cornell = load('dataset\cornell\cornell.mat');
texas = load('dataset\texas\texas.mat');
washington = load('dataset\washington\washington.mat');
wisconsin = load('dataset\wisconsin\wisconsin.mat');

% data = cornell;
% data = texas ;
% data = washington;
data = wisconsin;
[nVertex,nFeature] = size(data.F);
% data = polbook;
% [nVertex,~] = size(data.A);

% 得到相似度矩阵
A = data.A;                                         % 使用邻接矩阵构造相似度矩阵
% A = 10./(pdist2(data.F,data.F,'euclidean')+10);     % 使用欧拉距离构造相似度矩阵（全连接）
% A = 200./(pdist2(data.F,data.F,'cityblock')+200);   % 使用街区距离构造相似度矩阵（全连接）
% A = 1-pdist2(data.F,data.F,'hamming');              % 使用汉明距离构造相似度矩阵（全连接）
% A = 1-pdist2(data.F,data.F,'jaccard');              % 使用Jaccard相似度构造相似度矩阵（全连接）
% A = 1-pdist2(data.F,data.F,'cosine');               % 使用余弦相似度构造相似度矩阵（全连接）
% A = A.*data.A;                                      % 给原有的边根据相似度赋值
% top_k = 20;                                           % 取top_k相似度构造K近邻图
% [max_similarity, index_order] = sort(A,2,'descend');  % 对相似度进行排序
% A = zeros(nVertex);
% for i = 1:nVertex                                     % 构造K近邻图作为相似度矩阵
%     for j = 2:(top_k+1)
%         A(i,index_order(i,j)) = max_similarity(i,j);
%         A(index_order(i,j),i) = max_similarity(i,j);
%     end
% end
% figure;
% plot(graph(A));

% 降维后矩阵的维度（即社区个数）
k = 5;

% 初始化社区指示矩阵和基矩阵
U = rand(nVertex,k);     % 基矩阵
V = rand(nVertex,k);     % 社区指示矩阵

% 更新社区指示矩阵和基矩阵
max_iterator = 45;
for i = 1:max_iterator
    U = U.*(A*V)./(U*(V'*V));  % 更新基矩阵
    V = V.*(A'*U)./(V*(U'*U)); % 更新社区指示矩阵
end

% 根据社区指示矩阵确定类标
[~,label] = max(V,[],2);

% 画出最终结果
% final_graph = zeros(nVertex);
% for i = 1:nVertex
%     for j = (i+1):nVertex
%         if label(i) == label(j)
%             final_graph(i,j) = 1;
%             final_graph(j,i) = 1;
%         end
%     end
% end
% figure;
% plot(graph(final_graph));

% 评估聚类结果
result = ClusteringMeasure(label,data.label)