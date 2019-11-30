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
nDimension = 100;     % 降维后的维度
nCommunity = 5;      % 社区的数量
alpha = 0.1;
beta = 0.5;
lambda = 1e9;

% 获取图的邻接矩阵
A = data.A; 

% 计算一阶和二阶相似度矩阵S：S=S1+5*S2
S1 = A;                                                 % 计算一阶相似度
S2 = 1-pdist2(S1,S1,'cosine');                          % 计算二阶相似度
S = S1+5*S2;                                            % 计算矩阵S

% 计算矩阵B1：B1(i,j)=(k_i*k_j)/(2*e)
e = sum(sum(A))/2;                                      % 图中边的总数
k = sum(A,2);                                           % 图中各结点的度
B1 = (repmat(k,1,nVertex).*repmat(k',nVertex,1))/(2*e); % 计算矩阵B1

% 初始化
M = rand(nVertex,nDimension);                           % 初始化基矩阵
U = rand(nVertex,nDimension);                           % 初始化结点表示矩阵
C = rand(nCommunity,nDimension);                        % 初始化社区表示矩阵
H = rand(nVertex,nCommunity);                           % 初始化社区指示矩阵

% 优化目标函数
t = 1;                                                  % 记录迭代次数
threshold = 1e2;
objective_0 = norm(S-M*U','fro')^2 ...                  % 计算最初的目标函数
    +alpha*norm(H-U*C','fro')^2 ...
    -beta*trace(H'*(A-B1)*H)...
    +lambda*norm(H'*H-eye(nCommunity))^2;
objective = [objective_0];
while true
   % 更新矩阵M
   M = M.*((S*U)./(M*(U'*U)));
   
   % 更新矩阵U
   U = U.*((S'*M+alpha*H*C)./(U*((M'*M)+alpha*(C'*C))));
   
   % 更新矩阵C
   C = C.*((H'*U)./(C*(U'*U)));
   
   % 更新矩阵H
   delta = (2*beta*(B1*H)).*(2*beta*(B1*H))+16*lambda*(H*(H'*H))...
       .*(2*beta*A*H+2*alpha*U*C'+(4*lambda-2*alpha)*H);
   H = H.*sqrt((-2*beta*B1*H+sqrt(delta))./(8*lambda*H*(H'*H)));
   
   % 迭代次数加一
   t = t+1;
   
   % 计算目标函数的值
   objective_i = norm(S-M*U','fro')^2+alpha*norm(H-U*C','fro')^2 ...
       -beta*trace(H'*(A-B1)*H)+lambda*norm(H'*H-eye(nCommunity))^2;
   objective = [objective objective_i];
   
   % 目标函数收敛时停止更新
   if objective(t-1)-objective(t) < threshold
       break;
   end
   
   % 出现NAN时跳出循环
   if isnan(objective(t))
       break;
   end
end

% 多次执行分类任务，取ACC的平均值
% 调用了libsvm工具包
% SUM_ACC = 0;
% for i = 1:30
%     nTrain = round(0.8*nVertex);                               % 训练集比例为80%
%     TrainIndex = randperm(nVertex,nTrain);                     % 随机抽取80%数据为训练集
%     TestIndex = setdiff(1:nVertex,TrainIndex);                 % 剩余为训练集
%     Train = U(TrainIndex,:);                                   % 训练集样本
%     Test = U(TestIndex,:);                                     % 测试集样本
%     TrainLabel = data.label(TrainIndex);                       % 测试集的真正标签
%     TestLabel = data.label(TestIndex);                         % 测试集的真正标签
%     model = svmtrain(TrainLabel,Train,'-t 2 -c 1 -g 0.07');    % 训练svm模型
%     PredictLabel = svmpredict(TestLabel, Test, model);         % 使用训练好的svm模型预测
%     ACC = classificationACC(TestLabel,PredictLabel);           % 评估分类效果
%     SUM_ACC = SUM_ACC+ACC;
% end
% AVG_ACC = SUM_ACC/30