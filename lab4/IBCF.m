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

[Precision_11,Recall_11,F1Score_11,Loss_11,TrainRating_11,TestRating_11,Similarity_11,PredictedRating_11,Indicator_11] = IBFC_Function(train_small_1,'CS');
[Precision_12,Recall_12,F1Score_12,Loss_12,TrainRating_12,TestRating_12,Similarity_12,PredictedRating_12,Indicator_12] = IBFC_Function(train_small_1,'PC');
[Precision_13,Recall_13,F1Score_13,Loss_13,TrainRating_13,TestRating_13,Similarity_13,PredictedRating_13,Indicator_13] = IBFC_Function(train_small_1,'AC');

[Precision_21,Recall_21,F1Score_21,Loss_21,TrainRating_21,TestRating_21,Similarity_21,PredictedRating_21,Indicator_21] = IBFC_Function(train_small_2,'CS');
[Precision_22,Recall_22,F1Score_22,Loss_22,TrainRating_22,TestRating_22,Similarity_22,PredictedRating_22,Indicator_22] = IBFC_Function(train_small_2,'PC');
[Precision_23,Recall_23,F1Score_23,Loss_23,TrainRating_23,TestRating_23,Similarity_23,PredictedRating_23,Indicator_23] = IBFC_Function(train_small_2,'AC');

[Precision_31,Recall_31,F1Score_31,Loss_31,TrainRating_31,TestRating_31,Similarity_31,PredictedRating_31,Indicator_31] = IBFC_Function(movielens_100k,'CS');
[Precision_32,Recall_32,F1Score_32,Loss_32,TrainRating_32,TestRating_32,Similarity_32,PredictedRating_32,Indicator_32] = IBFC_Function(movielens_100k,'PC');
[Precision_33,Recall_33,F1Score_33,Loss_33,TrainRating_33,TestRating_33,Similarity_33,PredictedRating_33,Indicator_33] = IBFC_Function(movielens_100k,'AC');

[Precision_41,Recall_41,F1Score_41,Loss_41,TrainRating_41,TestRating_41,Similarity_41,PredictedRating_41,Indicator_41] = IBFC_Function(movielens_1m,'CS');
[Precision_42,Recall_42,F1Score_42,Loss_42,TrainRating_42,TestRating_42,Similarity_42,PredictedRating_42,Indicator_42] = IBFC_Function(movielens_1m,'PC');
[Precision_43,Recall_43,F1Score_43,Loss_43,TrainRating_43,TestRating_43,Similarity_43,PredictedRating_43,Indicator_43] = IBFC_Function(movielens_1m,'AC');

function [Precision,Recall,F1Score,Loss,TrainRating,TestRating,Similarity,PredictedRating,Indicator] = IBFC_Function(Data,SimilarityType)

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

% 根据训练集求项目间的相似度
Similarity = zeros(nItem);
if SimilarityType == 'CS'                        % 根据公式求余弦相似度
    for i = 1:nItem
        for j = i:nItem
            % 余弦相似度的计算公式
            Similarity(i,j) = dot(TrainRating(:,i),TrainRating(:,j))...
                /(norm(TrainRating(:,i))*norm(TrainRating(:,j)));
            if isnan(Similarity(i,j))
                Similarity(i,j) = 0;
            end
            Similarity(j,i) = Similarity(i,j);
        end
    end
elseif SimilarityType == 'PC'                    % 根据公式求皮尔逊相关系数
    for i = 1:nItem
        for j = i:nItem
            % 找到同时评分了两个项目的用户
            temp_index = (TrainRating(:,i)~=0)&(TrainRating(:,j)~=0);
            % 计算项目的相似度
            if numel(find(temp_index)) == 0      % 没有用户同时评分两个项目
                Similarity(i,j) = 0;             % 项目相似度置为0
                Similarity(j,i) = 0;
            else
                % 项目的评分向量
                item_i = TrainRating(temp_index,i);
                item_j = TrainRating(temp_index,j);
                % 项目的平均评分
                avg_vector_i = sum(item_i)/numel(find(temp_index));
                avg_vector_j = sum(item_j)/numel(find(temp_index));
                % 皮尔逊相关系数的计算公式
                Similarity(i,j) = dot(item_i-avg_vector_i,item_j-avg_vector_j)...
                    /(sqrt(sum((item_i-avg_vector_i).^2))*sqrt(sum((item_j-avg_vector_j).^2)));
                if isnan(Similarity(i,j))
                    Similarity(i,j) = 0;
                end
                Similarity(j,i) = Similarity(i,j);
            end
        end
    end
elseif SimilarityType == 'AC'                   % 根据公式求余弦性适应相似度
    for i = 1:nItem
        for j = i:nItem
            % 找到同时评分了两个项目的用户
            temp_index = (TrainRating(:,i)~=0)&(TrainRating(:,j)~=0);
            % 计算项目的相似度
            if numel(find(temp_index)) == 0     % 没有用户同时评分两个项目
                Similarity(i,j) = 0;            % 项目相似度置为0
                Similarity(j,i) = 0;
            else
                % 项目的评分向量
                item_i = TrainRating(temp_index,i);
                item_j = TrainRating(temp_index,j);
                % 用户的平均评分
                avg_user = sum(TrainRating(temp_index,:),2)...
                    ./sum(TrainRating(temp_index,:)~=0,2);
                % 余弦适应性相似度的计算公式
                Similarity(i,j) = dot(item_i-avg_user,item_j-avg_user)...
                    /(sqrt(sum((item_i-avg_user).^2))*sqrt(sum((item_j-avg_user).^2)));
                if isnan(Similarity(i,j))
                    Similarity(i,j) = 0;
                end
                Similarity(j,i) = Similarity(i,j);
            end
        end
    end
end

% 根据项目相似度预测评分
Loss = 0;                             % 损失函数
PredictedRating = zeros(nUser,nItem); % 预测评分矩阵
Indicator = zeros(nUser,nItem);       % 推荐指示矩阵
                                      % (1:测试集行为;2:推荐列表;3:重合部分)
for i = 1:nUser
    % 用户的平均评分
    avg_rating = sum(TrainRating(i,:))/sum(TrainRating(i,:)~=0);
    % 设定推荐给用户的评分阈值
    threshold = 3;                                % 默认阈值为3分
    if avg_rating ~= 0 && ~isnan(avg_rating)      % 若用户有评分记录
        threshold = avg_rating;                   % 评分记录的平均值作为阈值
    end
    for j = 1:nItem
        % 预测用户对没有评分的项目的评分
        if TrainRating(i,j) == 0
            % 计算预测的评分
            PredictedRating(i,j) = dot(Similarity(j,:),TrainRating(i,:))...
                /sum(Similarity(j,TrainRating(i,:)~=0)); 
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