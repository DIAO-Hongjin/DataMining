% 载入数据集
% 有属性的小数据集
cornell = load('dataset\cornell.mat');
texas = load('dataset\texas.mat');
washington = load('dataset\washington.mat');
wisconsin = load('dataset\wisconsin.mat');
% 有属性的大数据集
BlogCatalog = load('dataset\BlogCatalog.mat');
Flickr = load('dataset\Flickr.mat');

% cornell数据集
cornell_label = cornell.label;
cornell_representations = SDANE_Function(cornell,'cornell',0.2,3,8,0.1,0.1,0.001,0.01,200,50);
classification_measure1 = classification(cornell_representations,cornell_label);
clustering_measure1 = clustering(cornell_representations,cornell_label,5);

% cornell数据集（无属性，即alpha=1）
% cornell_label = cornell.label;
% cornell_representations = SDANE_Function(cornell,'cornell',1,0,8,0,0.1,0,0.005,0,50);
% classification_measure1 = classification(cornell_representations,cornell_label);
% clustering_measure1 = clustering(cornell_representations,cornell_label,5);

% cornell数据集（全属性，即alpha=0）
% cornell_label = cornell.label;
% cornell_representations = SDANE_Function(cornell,'cornell',0,3,0,0.1,0.1,0.001,0.01,200,50);
% classification_measure1 = classification(cornell_representations,cornell_label);
% clustering_measure1 = clustering(cornell_representations,cornell_label,5);

% texas数据集
texas_label = texas.label;
texas_representations = SDANE_Function(texas,'texas',0.2,3,8,0.1,0.1,0.001,0.01,200,50);
classification_measure2 = classification(texas_representations,texas_label);
clustering_measure2 = clustering(texas_representations,texas_label,5);

% texas数据集（无属性，即alpha=1）
% texas_label = texas.label;
% texas_representations = SDANE_Function(texas,'texas',1,0,8,0,0.1,0,0.005,0,50);
% classification_measure2 = classification(texas_representations,texas_label);
% clustering_measure2 = clustering(texas_representations,texas_label,5);

% texas数据集（全属性，即alpha=0）
% texas_label = texas.label;
% texas_representations = SDANE_Function(texas,'texas',0,3,0,0.1,0.1,0.001,0.01,200,50);
% classification_measure2 = classification(texas_representations,texas_label);
% clustering_measure2 = clustering(texas_representations,texas_label,5);

% washington数据集
washington_label = washington.label;
washington_representations = SDANE_Function(washington,'washington',0.2,3,8,0.1,0.1,0.001,0.01,200,50);
classification_measure3 = classification(washington_representations,washington_label);
clustering_measure3 = clustering(washington_representations,washington_label,5);

% washington数据集（无属性，即alpha=1）
% washington_label = washington.label;
% washington_representations = SDANE_Function(washington,'washington',1,0,8,0,0.1,0,0.005,0,50);
% classification_measure3 = classification(washington_representations,washington_label);
% clustering_measure3 = clustering(washington_representations,washington_label,5);

% washington数据集（全属性，即alpha=0）
% washington_label = washington.label;
% washington_representations = SDANE_Function(washington,'washington',0,3,0,0.1,0.1,0.001,0.01,200,50);
% classification_measure3 = classification(washington_representations,washington_label);
% clustering_measure3 = clustering(washington_representations,washington_label,5);

% wisconsin数据集
wisconsin_label = wisconsin.label;
wisconsin_representations = SDANE_Function(wisconsin,'wisconsin',0.2,3,8,0.1,0.1,0.001,0.01,200,50);
classification_measure4 = classification(wisconsin_representations,wisconsin_label);
clustering_measure4 = clustering(wisconsin_representations,wisconsin_label,5);

% wisconsin数据集（无属性，即alpha=1）
% wisconsin_label = wisconsin.label;
% wisconsin_representations = SDANE_Function(wisconsin,'wisconsin',1,0,8,0,0.1,0,0.005,0,50);
% classification_measure4 = classification(wisconsin_representations,wisconsin_label);
% clustering_measure4 = clustering(wisconsin_representations,wisconsin_label,5);

% wisconsin数据集（全属性，即alpha=0）
% wisconsin_label = wisconsin.label;
% wisconsin_representations = SDANE_Function(wisconsin,'wisconsin',0,3,0,0.1,0.1,0.001,0.01,200,50);
% classification_measure4 = classification(wisconsin_representations,wisconsin_label);
% clustering_measure4 = clustering(wisconsin_representations,wisconsin_label,5);

% BlogCatalog数据集
% BlogCatalog_label = BlogCatalog.Label;
% BlogCatalog_representations = SDANE_Function(BlogCatalog,'BlogCatalog',0.4,15,10,0.1,0.1,0.00005,0.005,500,100);
% classification_measure = classification(BlogCatalog_representations,BlogCatalog_label);
% clustering_measure = clustering(BlogCatalog_representations,BlogCatalog_label,6);

% BlogCatalog数据集（无属性，即alpha=1) 
% BlogCatalog_label = BlogCatalog.Label;
% BlogCatalog_representations = SDANE_Function(BlogCatalog,'BlogCatalog',1,0,10,0.1,0.1,0,0.0005,0,100);
% classification_measure = classification(BlogCatalog_representations,BlogCatalog_label);
% clustering_measure = clustering(BlogCatalog_representations,BlogCatalog_label,6);

% BlogCatalog数据集（全属性，即alpha=0）
% BlogCatalog_label = BlogCatalog.Label;
% BlogCatalog_representations = SDANE_Function(BlogCatalog,'BlogCatalog',0,15,0,0.1,0.1,0.00005,0.005,500,100);
% classification_measure = classification(BlogCatalog_representations,BlogCatalog_label);
% clustering_measure = clustering(BlogCatalog_representations,BlogCatalog_label,6);

% Flickr数据集
% Flickr_label = Flickr.Label;
% Flickr_representations = SDANE_Function(Flickr,'Flickr',0.4,30,15,0.1,0.1,0.0001,0.00001,500,100);
% classification_measure = classification(Flickr_representations,Flickr_label);
% clustering_measure = clustering(Flickr_representations,Flickr_label,9);

% Flickr数据集（无属性，即alpha=1)
% Flickr_label = Flickr.Label;
% Flickr_representations = SDANE_Function(Flickr,'Flickr',1,0,15,0,0.1,0,0.000005,0,100);
% classification_measure = classification(Flickr_representations,Flickr_label);
% clustering_measure = clustering(Flickr_representations,Flickr_label,9);

% Flickr数据集（全属性，即alpha=0）
% Flickr_label = Flickr.Label;
% Flickr_representations = SDANE_Function(Flickr,'Flickr',0,30,0,0.1,0.1,0.0001,0.00001,500,100);
% classification_measure = classification(Flickr_representations,Flickr_label);
% clustering_measure = clustering(Flickr_representations,Flickr_label,9);

% SDANE学习节点的嵌入表示
% 输入参数：数据集、数据集名称、超参数alpha、节点属性的非零元素权重、节点拓扑的非零元素权重、
% 节点属性特征提取的正则化参数、节点表示学习的正则化参数、节点属性特征提取的步长、节点表示的步长、
% 节点属性降维后的维度、节点表示的维度
function node_representations = SDANE_Function(data,name,alpha,beta1,beta2,lambda1,lambda2,eta1,eta2,K1,K2)

% 获得网络的拓扑结构、节点属性和节点标签
if strcmp(name,'BlogCatalog') || strcmp(name,'Flickr')
    network = data.Network;
    attribute = data.Attributes;
else
    network = data.A;
    attribute = data.F;
end

% 网络嵌入模型
if alpha == 1
    % 通过SDAE得到网络中节点的表示
    node_representations = SDAE_Function(network,beta2,lambda2,eta2,K2);
else
    % 通过SDAE得到降维后的节点属性
    attribute_representations = SDAE_Function(attribute,beta1,lambda1,eta1,K1);
    % 通过aSDAE得到网络中节点的表示
    node_representations = aSDAE_Function(attribute_representations,network,alpha,beta2,lambda2,eta2,K2);
end

end

% SDAE提取节点属性的特征（即降维）
% 输入参数：节点属性、非零元素权重、正则化系数、步长、降维后维度
function attribute_representations = SDAE_Function(attribute,beta,lambda,eta,K)

% % 目标函数
% L1 = [];

% 样本数
n_sample = size(attribute,1);

% 每层的维度
input_size = size(attribute,2);        % 输入层的维度
hidden_size = K;                       % 隐藏层的维度（即节点属性降维后的维度）
output_size = input_size;              % 输出层的维度

% 稀疏性加权矩阵
B1 = 1*(attribute==0)+beta*(attribute~=0);

% 初始化权重矩阵和偏置向量
r = sqrt(6)/sqrt(hidden_size+output_size+1);
U1 = rand(hidden_size,input_size)*2*r-r;     % 输入层到隐藏层之间的权重
U2 = rand(output_size,hidden_size)*2*r-r;    % 隐藏层到输出层之间的权重
b11 = zeros(hidden_size,1);                  % 隐藏层的偏置
b12 = zeros(output_size,1);                  % 输出层的偏置

% % 目标函数的初始值
% L1 = [L1 objective1(attribute,beta,lambda,U1,U2,b11,b12)];

% 逐层训练
% 第一层（反馈训练）
corrupt_attr = attribute.*(rand(n_sample,input_size)>0.3);    % 给输入数据添加噪声
% for i = 1:5000
for i = 1:10000
    % 随机梯度下降法
    index = randi(n_sample);
    % 前向传播
    % 隐藏层
    input11 = (U1*corrupt_attr(index,:)'+b11)';     % 输入
    output11 = tanh(input11);                       % 输出
    % 输出层
    input12 = (U2*output11'+b12)';                  % 输入
    output12 = input12;                             % 输出
    % 反向传导
    % 输出层
    delta2 = (output12-attribute(index,:)).*B1(index,:);    % 输出层的加权误差
    U2 = U2-eta*(delta2'*output11+lambda*U2);               % 更新权重
    b12 = b12-eta*delta2';                                  % 更新偏置
    % 隐藏层
    % delta1 = sum(U2.*(delta2'*(1-output11.^2)));            % 传导误差
    delta1 = ((U2'*delta2').*(1-output11.^2)')';            % 传导误差
    U1 = U1-eta*(delta1'*corrupt_attr(index,:)+lambda*U1);  % 更新权重
    b11 = b11-eta*delta1';                                  % 更新偏置
    
%     % 目标函数的变化情况
%     if mod(i,50) == 0
%         L1 = [L1 objective1(attribute,beta,lambda,U1,U2,b11,b12)];
%     end
end

% % 目标函数L1的变化情况
% figure;
% plot(1:numel(L1),L1);

% 第二层隐藏层的输出作为降维后的节点属性
attribute_representations = tanh(U1*attribute'+repmat(b11,1,n_sample))';

end

% 目标函数L1
function loss = objective1(attribute,beta,lambda,U1,U2,b11,b12)

% 样本数
n_sample = size(attribute,1);

% 稀疏性加权矩阵
B1 = 1*(attribute==0)+beta*(attribute~=0);

% 隐藏层
input1 = (U1*attribute'+repmat(b11,1,n_sample))';    % 输入
output1 = tanh(input1);                              % 输出
% 输出层
input2 = (U2*output1'+repmat(b12,1,n_sample))';      % 输入
output2 = input2;                                    % 输出

% 计算L1
loss = 0.5*norm((output2-attribute).*B1,'fro')^2+0.5*lambda*(norm(U1,'fro')^2+norm(U2,'fro')^2);

end

% aSDAE学习网络的节点表示
% 输入参数：节点属性、超参数alpha、非零元素权重、正则化系数、步长、嵌入表示的维度
function node_representations = aSDAE_Function(attribute_representations,network,alpha,beta,lambda,eta,K)

% % 目标函数
% L2 = [];

% 样本数
n_sample = size(network,1);
% 属性维度
n_attr_dim = size(attribute_representations,2);

% 每层的维度
input_size = size(network,2);           % 输入层的维度
hidden_size = K;                        % 隐藏层的维度（即节点嵌入表示的维度）
outputX_size = input_size;              % 输出层的维度
outputZ_size = n_attr_dim;

% 稀疏性加权矩阵
B2 = 1*(network==0)+beta*(network~=0);

% 初始化权重矩阵和偏置向量
r1 = sqrt(6)/sqrt(hidden_size+outputX_size+1);
r2 = sqrt(6)/sqrt(hidden_size+outputZ_size+1);
W1 = rand(hidden_size,input_size)*2*r1-r1;      % 输入层到隐藏层之间的权重
W2 = rand(outputX_size,hidden_size)*2*r1-r1;    % 隐藏层到输出层（重构邻接矩阵）之间的权重
V1 = rand(hidden_size,n_attr_dim)*2*r2-r2;      % 节点属性到隐藏层之间的权重
V2 = rand(outputZ_size,hidden_size)*2*r2-r2;    % 隐藏层到输出层（重构属性特征）之间的权重
b21 = zeros(hidden_size,1);                     % 隐藏层的偏置
b22X = zeros(outputX_size,1);                   % 输出层（重构邻接矩阵）的偏置
b22Z = zeros(outputZ_size,1);                   % 输出层（重构属性特征）的偏置

% % 目标函数的初始值
% L2 = [L2 objective2(attribute_representations,network,alpha,beta,lambda,V1,V2,W1,W2,b21,b22X,b22Z)];

% 反馈训练
corrupt_network = network.*(rand(n_sample,input_size)>0.3);    % 给输入数据添加噪声
% for i = 1:5000
for i = 1:10000
    % 随机梯度下降法
    index = randi(n_sample);
    % 前向传播
    % 隐藏层
    input1 = (W1*corrupt_network(index,:)'+V1*attribute_representations(index,:)'+b21)';    % 输入
    output1 = tanh(input1);                                                                 % 输出
    % 输出层
    input2X = (W2*output1'+b22X)';                                                          % 输入
    input2Z = (V2*output1'+b22Z)';
    output2X = sigmoid(input2X);                                                            % 输出
    output2Z = input2Z;  
   
    % 反向传导
    % 输出层
    delta2X = alpha*((output2X-network(index,:)).*B2(index,:)).*(output2X.*(1-output2X));    % 输出层的加权误差
    delta2Z = (1-alpha)*(output2Z-attribute_representations(index,:));
    W2 = W2-eta*(delta2X'*output1+lambda*W2);                                                % 更新权重
    V2 = V2-eta*(delta2Z'*output1+lambda*V2);
    b22X = b22X-eta*delta2X';                                                                % 更新偏置
    b22Z = b22Z-eta*delta2Z';
    % 隐藏层
%     delta1X = sum(W2.*(delta2X'*(1-output1.^2)));                                            % 传导误差
%     delta1Z = sum(V2.*(delta2Z'*(1-output1.^2)));
    delta1X = ((W2'*delta2X').*(1-output1.^2)')';                                            % 传导误差
    delta1Z = ((V2'*delta2Z').*(1-output1.^2)')';
    W1 = W1-eta*(delta1X'*corrupt_network(index,:)+lambda*W1);                               % 更新权重
    V1 = V1-eta*(delta1Z'*attribute_representations(index,:)+lambda*V1);
    b21 = b21-eta*(delta1X'+delta1Z');                                                       % 更新偏置
   
%     % 目标函数的变化情况
%     if mod(i,50) == 0
%         L2 = [L2 objective2(attribute_representations,network,alpha,beta,lambda,V1,V2,W1,W2,b21,b22X,b22Z)];
%     end
end

% % 目标函数L2的变化情况
% figure;
% plot(1:numel(L2),L2);

% 第二层隐藏层的输出作为降维后的节点属性
node_representations = tanh(W1*network'+V1*attribute_representations'+repmat(b21,1,n_sample))';

end

% 目标函数L2
function loss = objective2(attribute_representations,network,alpha,beta,lambda,V1,V2,W1,W2,b21,b22X,b22Z)

% 样本数
n_sample = size(network,1);

% 稀疏性加权矩阵
B2 = 1*(network==0)+beta*(network~=0);

% 隐藏层
input1 = (W1*network'+V1*attribute_representations'+repmat(b21,1,n_sample))';    % 输入
output1 = tanh(input1);                                                          % 输出
% 输出层
input2X = (W2*output1'+repmat(b22X,1,n_sample))';                                % 输入
input2Z = (V2*output1'+repmat(b22Z,1,n_sample))';
output2X = sigmoid(input2X);                                                     % 输出
output2Z = input2Z;                                    

% 计算L2
loss = 0.5*alpha*norm((output2X-network).*B2,'fro')^2 ...
    +0.5*(1-alpha)*norm((output2Z-attribute_representations),'fro')^2 ...
    +0.5*lambda*(norm(V1,'fro')^2+norm(V2,'fro')^2+norm(W1,'fro')^2+norm(W2,'fro')^2);

end

% 分类任务
function classification_measure = classification(node_representations,label)

% 样本数
n_sample = size(node_representations,1); 

classification_measure = 0;
for i = 1:5
    % 划分测试集与训练集
    test_index = i:5:n_sample;                          % 测试集样本
    train_index = setdiff(1:n_sample,test_index);       % 训练集样本
    test_set = node_representations(test_index,:);      % 测试集
    train_set = node_representations(train_index,:);    % 训练集
    test_label = label(test_index,:);                   % 测试集真正标签
    train_label = label(train_index,:);                 % 训练集真正标签
    
    % 调用libsvm工具箱进行分类
    model = svmtrain(train_label,train_set,'-t 2 -c 1 -g 0.07');                                      % 训练svm模型
    predict_label = svmpredict(test_label,test_set,model);                                            % 使用训练好的svm模型预测
    classification_measure = classification_measure + classificationACC(test_label,predict_label);    % 评估分类效果
end

classification_measure = classification_measure/5;      % 取平均值作为最终结果

end

% 聚类任务
function clustering_measure = clustering(node_representations,label,cluster_num)

clustering_measure = zeros(1,3);
for i = 1:5
    % 节点的嵌入表示执行K-Means聚类
    cluster_id = kmeans(node_representations,cluster_num);
    % 评估聚类结果
    clustering_measure = clustering_measure+ClusteringMeasure(cluster_id,label);
end

clustering_measure = clustering_measure/5;    % 取平均值作为最终结果

end

% 激活函数之一：sigmoid函数
function Y = sigmoid(X)
Y = sigmf(X,[1 0]);
end