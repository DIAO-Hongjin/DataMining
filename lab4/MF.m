% 数据集读取
fid_ml_1m = fopen('datasets\movielens\ml-1m\ratings.dat');
movielens_1m = textscan(fid_ml_1m,'%n %*n %n %*n %n %*n %*n','Delimiter', '::','headerlines', 1);
fclose(fid_ml_1m);
fid_ml_100k = fopen('datasets\movielens\ml-100k\ml-100k\u.data');
movielens_100k = textscan(fid_ml_100k,'%n %n %n %*n','Delimiter', '\t');
fclose(fid_ml_100k);
fid_small_1 = fopen('datasets\small\train_small.txt');
train_small_1 = textscan(fid_small_1,'%n %n %n');
fclose(fid_small_1);
fid_small_2 = fopen('datasets\small\train_small_2.txt');
train_small_2 = textscan(fid_small_2,'%n %n %n');
fclose(fid_small_2);

[Precision_1,Recall_1,F1Score_1,TestLoss_1,~,~,~,~,TrainLoss_1,~,~] = MF_Function(train_small_1,20,0.00005,0.2);
[Precision_2,Recall_2,F1Score_2,TestLoss_2,~,~,~,~,TrainLoss_2,~,~] = MF_Function(train_small_2,40,0.00005,0.2);
[Precision_3,Recall_3,F1Score_3,TestLoss_3,~,~,~,~,TrainLoss_3,~,~] = MF_Function(movielens_100k,40,0.00005,0.2);
[Precision_4,Recall_4,F1Score_4,TestLoss_4,~,~,~,~,TrainLoss_4,~,~] = MF_Function(movielens_1m,40,0.00005,0.2);

function [Precision,Recall,F1Score,TestLoss,TrainRating,TestRating,P,Q,TrainLoss,PredictedRating,Indicator] = MF_Function(Data,K,alpha,beta)

% 获取用户总数、项目总数和记录总数
nUser = max(Data{1});                                            % 用户总数
nItem = max(Data{2});                                            % 项目总数
nInfo = size(Data{1},1);                                         % 记录总数

% 划分训练集和测试集
nTest = round(0.2*nInfo);                        % 测试集比例为记录总数的20%
TestIndex = randperm(nInfo,nTest);               % 随机抽取20%的记录为测试集
TrainIndex = setdiff(1:nInfo,TestIndex);         % 剩余为训练集
% 将训练集中的评分信息转换为相应的评分矩阵
TrainRating = full(sparse(Data{1}(TrainIndex),Data{2}(TrainIndex),Data{3}(TrainIndex),nUser,nItem));
% 将测试集中的评分信息转换为相应的评分矩阵
TestRating = full(sparse(Data{1}(TestIndex),Data{2}(TestIndex),Data{3}(TestIndex),nUser,nItem));

% 进行矩阵分解
P = rand(nUser,K);                                % 用户因子矩阵P
Q = rand(K,nItem);                                % 商品因子矩阵Q
TrainIndicator = (TrainRating~=0);                % 训练集非零元素指示矩阵
ErrorRating = TrainRating-(P*Q).*TrainIndicator;  % 原始矩阵与重构矩阵的误差
TrainLoss = sum(sum(ErrorRating.^2));             % 训练损失函数（初始值）
% 通过梯度下降法更新用户因子矩阵和商品因子矩阵，直到损失函数收敛
for t = 1:10000
    % 更新P和Q
    for i = 1:nUser
        for j = 1:nItem
            if TrainIndicator(i,j)
                for k = 1:K
                    P(i,k) = P(i,k)+alpha*(2*ErrorRating(i,j)*Q(k,j)-beta*P(i,k));
                    Q(k,j) = Q(k,j)+alpha*(2*ErrorRating(i,j)*P(i,k)-beta*Q(k,j));
                end
            end
        end
    end
    % 重新计算误差
    ErrorRating = TrainRating-(P*Q).*TrainIndicator;
    % 重新计算损失函数
    TrainLoss = [TrainLoss sum(sum(ErrorRating.^2));];
    % 损失函数收敛时停止迭代
    if TrainLoss(end-1)-TrainLoss(end) <= 0.001
        break;
    end
end

% 根据用户因子矩阵和项目因子矩阵的乘积预测评分
TestLoss = 0;                         % 损失函数
PredictedRating = P*Q;                % 预测评分矩阵
Indicator = zeros(nUser,nItem);       % 推荐指示矩阵
                                      % (1:测试集行为;2:推荐列表;3:重合部分)
for i = 1:nUser
    % 用户评价过的商品的集合
    pass_item = TrainRating(i,:)~=0;              % 历史评分项目
    pass_rating = TrainRating(i,pass_item);       % 历史评分记录
    % 用户的平均评分
    avg_rating = sum(pass_rating)/numel(pass_rating);
    % 设定推荐给用户的评分阈值
    threshold = 3;                                % 默认阈值为3分
    if numel(pass_rating) ~= 0                    % 若用户有评分记录
        threshold = avg_rating;                   % 评分记录的平均值作为阈值
    end
    for j = 1:nItem
        % 限制预测的评分的范围
        if isnan(PredictedRating(i,j)) || isinf(PredictedRating(i,j))
            PredictedRating(i,j) = 0;
        elseif PredictedRating(i,j) > 5
            PredictedRating(i,j) = 5;
        elseif PredictedRating(i,j) < 0
            PredictedRating(i,j) = 0;
        end
        % 分析用户没有评分的项目的预测评分
        if TrainRating(i,j) == 0
            % 若预测评分大于评分阈值，则推荐给该用户，在指示矩阵上记录
            if PredictedRating(i,j) >= threshold
                Indicator(i,j) = Indicator(i,j)+2;
            end
            % 若测试集上存在该行为，则计算损失函数，并在指示矩阵上记录
            if TestRating(i,j) ~= 0
                TestLoss = TestLoss+power(TestRating(i,j)-PredictedRating(i,j),2);
                Indicator(i,j) = Indicator(i,j)+1;
            end
        end
    end
end

% 计算评价指标
TestLoss = sqrt(TestLoss/nTest);                       % 计算测试集损失函数
Precision = sum(sum(Indicator==3))...                  % 计算准确率
    /sum(sum(Indicator==3|Indicator==2));
Recall = sum(sum(Indicator==3))...                     % 计算召回率
    /sum(sum(Indicator==3|Indicator==1));
F1Score = (2*Precision*Recall)/(Precision+Recall);     % 计算F1值

end