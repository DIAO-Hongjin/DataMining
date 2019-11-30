cornell = load('dataset\cornell\cornell.mat');
texas = load('dataset\texas\texas.mat');
washington = load('dataset\washington\washington.mat');
wisconsin = load('dataset\wisconsin\wisconsin.mat');

% data = cornell;
% data = texas ;
% data = washington;
data = wisconsin;

% 获取输入图的顶点数量和属性维度
[nVertex,nFeature] = size(data.F);

% 参数设置
d = 100;
alpha1 = 2;
alpha2 = 15;
epsilon = 0.1;
delta1 = 0.5;
delta2 = 1.5;

% 输入图中标签种类的数量
nCommunity = 5;

% 划分训练集和测试集
nTrain = round(0.8*nVertex);                       % 训练集比例为80%
TrainIndex = randperm(nVertex,nTrain);             % 随机抽取80%数据为训练集
TestIndex = setdiff(1:nVertex,TrainIndex);         % 剩余为训练集
G_Train = data.A(TrainIndex,TrainIndex);           % 训练集的邻接矩阵
A_Train = data.F(TrainIndex,:);                    % 训练集的节点属性
Y_Train = zeros(nTrain,nCommunity);                % 训练集的标签矩阵
for i = 1:nTrain
    Y_Train(i,data.label(TrainIndex(i))) = 1;
end
G_Test = data.A(TestIndex,TestIndex);              % 测试集的邻接矩阵
A_Test = data.F(TestIndex,:);                      % 测试集的节点属性
Y_Test = zeros(nVertex-nTrain,nCommunity);         % 测试集的标签矩阵
for i = 1:(nVertex-nTrain)
    Y_Test(i,data.label(TestIndex(i))) = 1;
end

% 计算图的拓扑结构的相似度矩阵：基于二阶相似度
S_G = zeros(nTrain); 
for i = 1:nTrain     
    for j = 1:nTrain 
        % 如果一对结点中有至少一个孤立点，则二阶相似度为0
        % 否则这对结点的二阶相似度为余弦相似度
        if norm(G_Train(i,:)) == 0 || norm(G_Train(j,:)) == 0
            S_G(i,j) = 0;
        else
            S_G(i,j) = dot(G_Train(i,:),G_Train(j,:))...
                /(norm(G_Train(i,:))*norm(G_Train(j,:)));
        end
    end
end

% 计算图的节点属性的相似度矩阵:基于余弦相似度
S_A = 1-pdist2(A_Train,A_Train,'cosine'); 

% 计算图的节点标签的相似度矩阵：基于余弦相似度
S_Y = 1-pdist2(Y_Train*Y_Train',Y_Train*Y_Train','cosine'); 

% 分别计算拓扑结构、节点属性和节点标签的拉普拉斯矩阵
D_G = diag(sum(S_G,2));                    % 计算拓扑结构的相似度矩阵的度矩阵
L_G = (D_G^-0.5)*S_G*(D_G^-0.5);           % 构造拓扑结构的拉普拉斯矩阵
L_G(isnan(L_G)) = 0;                       % 处理可能出现的NAN
L_G = 0.5*(L_G+L_G');                      % 处理为对称矩阵
D_A = diag(sum(S_A,2));                    % 计算节点属性的相似度矩阵的度矩阵
L_A = (D_A^-0.5)*S_A*(D_A^-0.5);           % 构造节点属性的拉普拉斯矩阵
L_A(isnan(L_A)) = 0;                       % 处理可能出现的NAN
L_A = 0.5*(L_A+L_A');                      % 处理为对称矩阵
D_Y = diag(sum(S_Y,2));                    % 计算节点标签的相似度矩阵的度矩阵
L_Y = (D_Y^-0.5)*S_Y*(D_Y^-0.5);           % 构造节点标签的拉普拉斯矩阵
L_Y(isnan(L_Y)) = 0;                       % 处理可能出现的NAN
L_Y = 0.5*(L_Y+L_Y');                      % 处理为对称矩阵

% 初始化表示矩阵
U_G = zeros(nTrain,d);                     % 拓扑结构的潜在表示
U_A = zeros(nTrain,d);                     % 节点属性的潜在表示
U_Y = zeros(nTrain,d);                     % 节点标签的潜在表示
H_Train = zeros(nTrain,d);                 % 训练集最终嵌入结果

% 优化目标函数
t = 1;                                     % 记录迭代次数
objective_0 = trace(U_G'*L_G*U_G)...       % 计算最初的目标函数
    +alpha1*trace(U_A'*L_A*U_A)...
    +alpha1*trace(U_A'*(U_G*U_G')*U_A)...
    +alpha2*trace(U_Y'*(L_Y+U_G*U_G')*U_Y)...
    +trace(U_G'*(H_Train*H_Train')*U_G)...
    +trace(U_A'*(H_Train*H_Train')*U_A)...
    +trace(U_Y'*(H_Train*H_Train')*U_Y);
objective = [objective_0];
while true                                 % 迭代更新各个变量
    % 更新U_G
    M1 = L_G+alpha1*(U_A*U_A')+alpha2*(U_Y*U_Y')+(H_Train*H_Train');
    [U_G,~] = eigs(M1,d);
    
    % 更新U_A
    M2 = alpha1*L_A+alpha1*(U_G*U_G')+(H_Train*H_Train');
    [U_A,~] = eigs(M2,d);
    
    % 更新U_Y
    M3 = alpha2*L_Y+alpha2*(U_G*U_G')+(H_Train*H_Train');
    [U_Y,~] = eigs(M3,d);
    
    % 更新H
    M4 = (U_G*U_G')+(U_A*U_A')+(U_Y*U_Y');
    [H_Train,~] = eigs(M4,d);
    
    % 迭代次数加一
    t = t+1;
    
    % 计算目标函数
    objective_t = trace(U_G'*L_G*U_G)...       
        +alpha1*trace(U_A'*L_A*U_A)...
        +alpha1*trace(U_A'*(U_G*U_G')*U_A)...
        +alpha2*trace(U_Y'*(L_Y+U_G*U_G')*U_Y)...
        +trace(U_G'*(H_Train*H_Train')*U_G)...
        +trace(U_A'*(H_Train*H_Train')*U_A)...
        +trace(U_Y'*(H_Train*H_Train')*U_Y);
    objective = [objective objective_t];
    
    % 目标函数收敛时停止更新 
    if objective(t)-objective(t-1) < epsilon
        break;
    end
end

% 得到测试集的最终嵌入表示
G1 = data.A(TrainIndex,:);                      % 训练集与所有节点的邻接矩阵
G2 = data.A(TestIndex,:);                       % 测试集与所有节点的邻接矩阵
H_Test = delta1*(G2*pinv(pinv(H_Train)*G1))...  % 测试集的嵌入表示
    +delta2*(A_Test*pinv(pinv(H_Train)*A_Train));

% 调用了libsvm工具包执行分类任务
% TrainLabel = data.label(TrainIndex);                       % 测试集的真正标签
% TestLabel = data.label(TestIndex);                         % 测试集的真正标签
% model = svmtrain(TrainLabel,H_Train,'-t 2 -c 1 -g 0.07');  % 训练svm模型
% PredictLabel = svmpredict(TestLabel, H_Test, model);       % 使用训练好的svm模型预测
% ACC = classificationACC(TestLabel,PredictLabel)            % 评估分类效果