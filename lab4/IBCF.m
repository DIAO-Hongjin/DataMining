% ���ݼ���ȡ
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

% ��ȡ�û���������Ŀ�����ͼ�¼����
nUser = max(Data{1});                                            % �û�����
nItem = max(Data{2});                                            % ��Ŀ����
nInfo = size(Data{1},1);                                         % ��¼����

% ����ѵ�����Ͳ��Լ�
nTest = round(0.2*nInfo);                        % ���Լ�����Ϊ��¼������20%
TestIndex = randperm(nInfo,nTest);               % �����ȡ20%�ļ�¼Ϊ���Լ�
TrainIndex = setdiff(1:nInfo,TestIndex);         % ʣ��Ϊѵ����
% ��ѵ�����е�������Ϣת��Ϊ��Ӧ�����־���
TrainRating = full(sparse(Data{1}(TrainIndex),Data{2}(TrainIndex),Data{3}(TrainIndex),nUser,nItem));
% �����Լ��е�������Ϣת��Ϊ��Ӧ�����־���
TestRating = full(sparse(Data{1}(TestIndex),Data{2}(TestIndex),Data{3}(TestIndex),nUser,nItem));

% ����ѵ��������Ŀ������ƶ�
Similarity = zeros(nItem);
if SimilarityType == 'CS'                        % ���ݹ�ʽ���������ƶ�
    for i = 1:nItem
        for j = i:nItem
            % �������ƶȵļ��㹫ʽ
            Similarity(i,j) = dot(TrainRating(:,i),TrainRating(:,j))...
                /(norm(TrainRating(:,i))*norm(TrainRating(:,j)));
            if isnan(Similarity(i,j))
                Similarity(i,j) = 0;
            end
            Similarity(j,i) = Similarity(i,j);
        end
    end
elseif SimilarityType == 'PC'                    % ���ݹ�ʽ��Ƥ��ѷ���ϵ��
    for i = 1:nItem
        for j = i:nItem
            % �ҵ�ͬʱ������������Ŀ���û�
            temp_index = (TrainRating(:,i)~=0)&(TrainRating(:,j)~=0);
            % ������Ŀ�����ƶ�
            if numel(find(temp_index)) == 0      % û���û�ͬʱ����������Ŀ
                Similarity(i,j) = 0;             % ��Ŀ���ƶ���Ϊ0
                Similarity(j,i) = 0;
            else
                % ��Ŀ����������
                item_i = TrainRating(temp_index,i);
                item_j = TrainRating(temp_index,j);
                % ��Ŀ��ƽ������
                avg_vector_i = sum(item_i)/numel(find(temp_index));
                avg_vector_j = sum(item_j)/numel(find(temp_index));
                % Ƥ��ѷ���ϵ���ļ��㹫ʽ
                Similarity(i,j) = dot(item_i-avg_vector_i,item_j-avg_vector_j)...
                    /(sqrt(sum((item_i-avg_vector_i).^2))*sqrt(sum((item_j-avg_vector_j).^2)));
                if isnan(Similarity(i,j))
                    Similarity(i,j) = 0;
                end
                Similarity(j,i) = Similarity(i,j);
            end
        end
    end
elseif SimilarityType == 'AC'                   % ���ݹ�ʽ����������Ӧ���ƶ�
    for i = 1:nItem
        for j = i:nItem
            % �ҵ�ͬʱ������������Ŀ���û�
            temp_index = (TrainRating(:,i)~=0)&(TrainRating(:,j)~=0);
            % ������Ŀ�����ƶ�
            if numel(find(temp_index)) == 0     % û���û�ͬʱ����������Ŀ
                Similarity(i,j) = 0;            % ��Ŀ���ƶ���Ϊ0
                Similarity(j,i) = 0;
            else
                % ��Ŀ����������
                item_i = TrainRating(temp_index,i);
                item_j = TrainRating(temp_index,j);
                % �û���ƽ������
                avg_user = sum(TrainRating(temp_index,:),2)...
                    ./sum(TrainRating(temp_index,:)~=0,2);
                % ������Ӧ�����ƶȵļ��㹫ʽ
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

% ������Ŀ���ƶ�Ԥ������
Loss = 0;                             % ��ʧ����
PredictedRating = zeros(nUser,nItem); % Ԥ�����־���
Indicator = zeros(nUser,nItem);       % �Ƽ�ָʾ����
                                      % (1:���Լ���Ϊ;2:�Ƽ��б�;3:�غϲ���)
for i = 1:nUser
    % �û���ƽ������
    avg_rating = sum(TrainRating(i,:))/sum(TrainRating(i,:)~=0);
    % �趨�Ƽ����û���������ֵ
    threshold = 3;                                % Ĭ����ֵΪ3��
    if avg_rating ~= 0 && ~isnan(avg_rating)      % ���û������ּ�¼
        threshold = avg_rating;                   % ���ּ�¼��ƽ��ֵ��Ϊ��ֵ
    end
    for j = 1:nItem
        % Ԥ���û���û�����ֵ���Ŀ������
        if TrainRating(i,j) == 0
            % ����Ԥ�������
            PredictedRating(i,j) = dot(Similarity(j,:),TrainRating(i,:))...
                /sum(Similarity(j,TrainRating(i,:)~=0)); 
            % ����Ԥ������ֵķ�Χ
            if isnan(PredictedRating(i,j)) || isinf(PredictedRating(i,j))
                PredictedRating(i,j) = 0;
            elseif PredictedRating(i,j) > 5
                PredictedRating(i,j) = 5;
            elseif PredictedRating(i,j) < 0
                PredictedRating(i,j) = 0;
            end
            % ��Ԥ�����ִ���������ֵ�����Ƽ������û�����ָʾ�����ϼ�¼
            if PredictedRating(i,j) >= threshold
                Indicator(i,j) = Indicator(i,j)+2;
            end
            % �����Լ��ϴ��ڸ���Ϊ���������ʧ����������ָʾ�����ϼ�¼
            if TestRating(i,j) ~= 0
                Loss = Loss+power(TestRating(i,j)-PredictedRating(i,j),2);
                Indicator(i,j) = Indicator(i,j)+1;
            end
        end
    end
end

% ��������ָ��
Loss = sqrt(Loss/nTest);                                     % ������ʧ����
Precision = sum(sum(Indicator==3))...                        % ����׼ȷ��
    /sum(sum(Indicator==3|Indicator==2));
Recall = sum(sum(Indicator==3))...                           % �����ٻ���
    /sum(sum(Indicator==3|Indicator==1));
F1Score = (2*Precision*Recall)/(Precision+Recall);           % ����F1ֵ

end