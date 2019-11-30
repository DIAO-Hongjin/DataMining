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

[Precision_1,Recall_1,F1Score_1,Loss_1,TrainRating_1,TestRating_1,Deviation_1,PredictedRating_1,Indicator_1] = SlopeOne_Function(train_small_1);
[Precision_2,Recall_2,F1Score_2,Loss_2,TrainRating_2,TestRating_2,Deviation_2,PredictedRating_2,Indicator_2] = SlopeOne_Function(train_small_2);
[Precision_3,Recall_3,F1Score_3,Loss_3,TrainRating_3,TestRating_3,Deviation_3,PredictedRating_3,Indicator_3] = SlopeOne_Function(movielens_100k);
[Precision_4,Recall_4,F1Score_4,Loss_4,TrainRating_4,TestRating_4,Deviation_4,PredictedRating_4,Indicator_4] = SlopeOne_Function(movielens_1m);

function [Precision,Recall,F1Score,Loss,TrainRating,TestRating,Deviation,PredictedRating,Indicator] = SlopeOne_Function(Data)

% 获取用户总数、项目总数和记录总数
nUser = max(Data{1});                                            % 用户总数
nItem = max(Data{2});                                            % 项目总数
nInfo = size(Data{1},1);                                           % 记录总数

% 划分训练集和测试集
nTest = round(0.2*nInfo);                        % 测试集比例为记录总数的20%
TestIndex = randperm(nInfo,nTest);               % 随机抽取20%的记录为测试集
TrainIndex = setdiff(1:nInfo,TestIndex);         % 剩余为训练集
% 将训练集中的评分信息转换为相应的评分矩阵
TrainRating = full(sparse(Data{1}(TrainIndex),Data{2}(TrainIndex),Data{3}(TrainIndex),nUser,nItem));
% 将测试集中的评分信息转换为相应的评分矩阵
TestRating = full(sparse(Data{1}(TestIndex),Data{2}(TestIndex),Data{3}(TestIndex),nUser,nItem));

% 计算商品之间的评分偏差
Deviation = zeros(nItem);                        % 商品之间的评分偏差
for i = 1:nItem
    for j = (i+1):nItem
        % 找到同时评分了两个项目的用户
        temp_index = (TrainRating(:,i)~=0)&(TrainRating(:,j)~=0);
        % 项目的评分向量
        item_i = TrainRating(temp_index,i);
        item_j = TrainRating(temp_index,j);
        % 计算评分偏差
        Deviation(i,j) = sum(item_i-item_j)/numel(find(temp_index));
        if isnan(Deviation(i,j))
            Deviation(i,j) = 0;
        end
        Deviation(j,i) = -Deviation(i,j);
    end
end

% 根据项目评分偏差和用户历史纪录预测评分
Loss = 0;                             % 损失函数
PredictedRating = zeros(nUser,nItem); % 预测评分矩阵
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
        % 预测用户对没有评分的项目的评分
        if TrainRating(i,j) == 0
            % 计算预测的评分
            PredictedRating(i,j) = sum(Deviation(j,pass_item)+pass_rating)/numel(pass_rating);
            % 限制预测的评分的范围
            if isnan(PredictedRating(i,j)) || isinf(PredictedRating(i,j))
                PredictedRating(i,j) = 0;
            elseif PredictedRating(i,j) > 5
                PredictedRating(i,j) = 5;
            elseif PredictedRating(i,j) < 0
                PredictedRating(i,j) = 0;
            end
            % 若预测评分大于评分阈值，则推荐给该用户，在指示矩阵上记录
            if PredictedRating(i,j) >= threshold
                Indicator(i,j) = Indicator(i,j)+2;
            end
            % 若测试集上存在该行为，则计算损失函数，并在指示矩阵上记录
            if TestRating(i,j) ~= 0
                Loss = Loss+power(TestRating(i,j)-PredictedRating(i,j),2);
                Indicator(i,j) = Indicator(i,j)+1;
            end
        end
    end
end

% 计算评价指标
Loss = sqrt(Loss/nTest);                                     % 计算损失函数
Precision = sum(sum(Indicator==3))...                        % 计算准确率
    /sum(sum(Indicator==3|Indicator==2));
Recall = sum(sum(Indicator==3))...                           % 计算召回率
    /sum(sum(Indicator==3|Indicator==1));
F1Score = (2*Precision*Recall)/(Precision+Recall);           % 计算F1值

end