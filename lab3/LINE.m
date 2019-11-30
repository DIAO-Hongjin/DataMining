cornell = load('dataset\cornell\cornell.mat');
texas = load('dataset\texas\texas.mat');
washington = load('dataset\washington\washington.mat');
wisconsin = load('dataset\wisconsin\wisconsin.mat');

data = cornell;
% data = texas ;
% data = washington;
% data = wisconsin;

% 获取输入图的顶点数量和属性维度
[nVertex,nFeature] = size(data.F);

% 获取输入图的所有边以及边的总数
[source,target] = find(data.A);                 % 得到图中所有边的始点和终点
nEdge = numel(source);                          % 边的总数

% 参数设置
d = 60;
rho_init = 0.025;
K = 5;
T = 10000;
M = 1e8;

% 初始化
U1 = rand(nVertex,d);          % 基于一阶相似度的嵌入表示
U2 = rand(nVertex,d);          % 基于二阶相似度的嵌入表示
U_ = rand(nVertex,d);          % 基于二阶相似度时节点作为邻居的表达

% 计算每个节点的长度(用于负采样)
Len = sum(data.A,2).^(3/4);
per_len = sum(Len)/M;

% 优化目标函数O1
rho = rho_init;
AVG_ACC1 = [];
for i = 1:T
    % 随机梯度下降：随机选取一条边
    current_edge = randi(nEdge);
    vi = source(current_edge);
    vj = target(current_edge);
    
    % 负采样
    vk = zeros(K,1);
    pos = randperm(M,K);
    for j = 1:K                   % 负采样K个点
        pointer = pos(j)*per_len;
        for x = 1:nVertex
            pointer = pointer-Len(x);
            if pointer < 0
                vk(j) = x;
                break;
            end
        end
    end
    
    % 求梯度及更新
    delta_uk = zeros(K,d);
    second = 0;
    for k = 1:K
        second = second...
            +U1(vk(k),:)*sigmoid(dot(U1(vk(k),:)',U1(vi,:)));
        delta_uk(k,:) = U1(vi,:)...             % 对uk求梯度
            *sigmoid(dot(U1(vk(k),:)',U1(vi,:)));
    end
    delta_ui = -(U1(vj,:)...                    % 对ui求梯度
        -U1(vj,:)*sigmoid(dot(U1(vj,:)',U1(vi,:)))-second);
    delta_uj = -U1(vi,:)...                     % 对uj求梯度
        *(1-sigmoid(dot(U1(vj)',U1(vi))));
    U1(vk,:) = U1(vk,:)-rho*delta_uk;           % 更新u
    U1(vi,:) = U1(vi,:)-rho*delta_ui;           % 更新ui
    U1(vj,:) = U1(vj,:)-rho*delta_uj;           % 更新uj
    
    % 学习率动态减小
    rho = rho_init*(1-i/T);
end

% 优化目标函数O2
rho = rho_init;
AVG_ACC2 = [];
for i = 1:T
    % 随机梯度下降：随机选取一条边
    current_edge = randi(nEdge);
    vi = source(current_edge);
    vj = target(current_edge);
    
    % 负采样
    vk = zeros(K,1);
    pos = randperm(M,K);
    for j = 1:K                   % 负采样K个点
        pointer = pos(j)*per_len;
        for x = 1:nVertex
            pointer = pointer-Len(x);
            if pointer < 0
                vk(j) = x;
                break;
            end
        end
    end
    
    % 求梯度及更新
    delta_uk = zeros(K,d);
    second = 0;
    for k = 1:K
        second = second...
            +U_(vk(k),:)*sigmoid(dot(U_(vk(k),:)',U2(vi,:)));
        delta_uk(k,:) = U2(vi,:)...             % 对uk'求梯度
            *sigmoid(dot(U_(vk(k),:)',U2(vi,:)));
    end
    delta_ui = -(U_(vj,:)...                    % 对ui求梯度
        -U_(vj,:)*sigmoid(dot(U_(vj,:)',U2(vi,:)))-second);
    delta_uj = -U2(vi,:)...                     % 对uj'求梯度
        *(1-sigmoid(dot(U_(vj)',U2(vi))));
    U_(vk,:) = U_(vk,:)-rho*delta_uk;           % 更新uk'
    U2(vi,:) = U2(vi,:)-rho*delta_ui;           % 更新ui
    U_(vj,:) = U_(vj,:)-rho*delta_uj;           % 更新uj'
    
    % 学习率动态减小
    rho = rho_init*(1-i/T);
end

% 多次执行分类任务，取ACC的平均值
% 调用了libsvm工具包

% % 基于一阶相似度
% SUM_ACC1 = 0;
% for i = 1:30
%     nTrain = round(0.8*nVertex);                               % 训练集比例为80%
%     TrainIndex = randperm(nVertex,nTrain);                     % 随机抽取80%数据为训练集
%     TestIndex = setdiff(1:nVertex,TrainIndex);                 % 剩余为训练集
%     Train = U1(TrainIndex,:);                                  % 训练集样本
%     Test = U1(TestIndex,:);                                    % 测试集样本
%     TrainLabel = data.label(TrainIndex);                       % 测试集的真正标签
%     TestLabel = data.label(TestIndex);                         % 测试集的真正标签
%     model = svmtrain(TrainLabel,Train,'-t 2 -c 1 -g 0.07');    % 训练svm模型
%     PredictLabel = svmpredict(TestLabel, Test, model);         % 使用训练好的svm模型预测
%     ACC1 = classificationACC(TestLabel,PredictLabel);          % 评估分类效果
%     SUM_ACC1 = SUM_ACC1+ACC1;
% end
% AVG_ACC1 = SUM_ACC1/30;
% 
% % 基于二阶相似度
% SUM_ACC2 = 0;
% for i = 1:30
%     nTrain = round(0.8*nVertex);                               % 训练集比例为80%
%     TrainIndex = randperm(nVertex,nTrain);                     % 随机抽取80%数据为训练集
%     TestIndex = setdiff(1:nVertex,TrainIndex);                 % 剩余为训练集
%     Train = U2(TrainIndex,:);                                  % 训练集样本
%     Test = U2(TestIndex,:);                                    % 测试集样本
%     TrainLabel = data.label(TrainIndex);                       % 测试集的真正标签
%     TestLabel = data.label(TestIndex);                         % 测试集的真正标签
%     model = svmtrain(TrainLabel,Train,'-t 2 -c 1 -g 0.07');    % 训练svm模型
%     PredictLabel = svmpredict(TestLabel, Test, model);         % 使用训练好的svm模型预测
%     ACC2 = classificationACC(TestLabel,PredictLabel);          % 评估分类效果
%     SUM_ACC2 = SUM_ACC2+ACC2;
% end
% AVG_ACC2 = SUM_ACC2/30;
% 
% [AVG_ACC1 AVG_ACC2]

function result = sigmoid(x)
result = 1/(1+exp(-x));
end