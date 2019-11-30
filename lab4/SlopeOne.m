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

[Precision_1,Recall_1,F1Score_1,Loss_1,TrainRating_1,TestRating_1,Deviation_1,PredictedRating_1,Indicator_1] = SlopeOne_Function(train_small_1);
[Precision_2,Recall_2,F1Score_2,Loss_2,TrainRating_2,TestRating_2,Deviation_2,PredictedRating_2,Indicator_2] = SlopeOne_Function(train_small_2);
[Precision_3,Recall_3,F1Score_3,Loss_3,TrainRating_3,TestRating_3,Deviation_3,PredictedRating_3,Indicator_3] = SlopeOne_Function(movielens_100k);
[Precision_4,Recall_4,F1Score_4,Loss_4,TrainRating_4,TestRating_4,Deviation_4,PredictedRating_4,Indicator_4] = SlopeOne_Function(movielens_1m);

function [Precision,Recall,F1Score,Loss,TrainRating,TestRating,Deviation,PredictedRating,Indicator] = SlopeOne_Function(Data)

% ��ȡ�û���������Ŀ�����ͼ�¼����
nUser = max(Data{1});                                            % �û�����
nItem = max(Data{2});                                            % ��Ŀ����
nInfo = size(Data{1},1);                                           % ��¼����

% ����ѵ�����Ͳ��Լ�
nTest = round(0.2*nInfo);                        % ���Լ�����Ϊ��¼������20%
TestIndex = randperm(nInfo,nTest);               % �����ȡ20%�ļ�¼Ϊ���Լ�
TrainIndex = setdiff(1:nInfo,TestIndex);         % ʣ��Ϊѵ����
% ��ѵ�����е�������Ϣת��Ϊ��Ӧ�����־���
TrainRating = full(sparse(Data{1}(TrainIndex),Data{2}(TrainIndex),Data{3}(TrainIndex),nUser,nItem));
% �����Լ��е�������Ϣת��Ϊ��Ӧ�����־���
TestRating = full(sparse(Data{1}(TestIndex),Data{2}(TestIndex),Data{3}(TestIndex),nUser,nItem));

% ������Ʒ֮�������ƫ��
Deviation = zeros(nItem);                        % ��Ʒ֮�������ƫ��
for i = 1:nItem
    for j = (i+1):nItem
        % �ҵ�ͬʱ������������Ŀ���û�
        temp_index = (TrainRating(:,i)~=0)&(TrainRating(:,j)~=0);
        % ��Ŀ����������
        item_i = TrainRating(temp_index,i);
        item_j = TrainRating(temp_index,j);
        % ��������ƫ��
        Deviation(i,j) = sum(item_i-item_j)/numel(find(temp_index));
        if isnan(Deviation(i,j))
            Deviation(i,j) = 0;
        end
        Deviation(j,i) = -Deviation(i,j);
    end
end

% ������Ŀ����ƫ����û���ʷ��¼Ԥ������
Loss = 0;                             % ��ʧ����
PredictedRating = zeros(nUser,nItem); % Ԥ�����־���
Indicator = zeros(nUser,nItem);       % �Ƽ�ָʾ����
                                      % (1:���Լ���Ϊ;2:�Ƽ��б�;3:�غϲ���)
for i = 1:nUser
    % �û����۹�����Ʒ�ļ���
    pass_item = TrainRating(i,:)~=0;              % ��ʷ������Ŀ
    pass_rating = TrainRating(i,pass_item);       % ��ʷ���ּ�¼
    % �û���ƽ������
    avg_rating = sum(pass_rating)/numel(pass_rating);
    % �趨�Ƽ����û���������ֵ
    threshold = 3;                                % Ĭ����ֵΪ3��
    if numel(pass_rating) ~= 0                    % ���û������ּ�¼
        threshold = avg_rating;                   % ���ּ�¼��ƽ��ֵ��Ϊ��ֵ
    end
    for j = 1:nItem
        % Ԥ���û���û�����ֵ���Ŀ������
        if TrainRating(i,j) == 0
            % ����Ԥ�������
            PredictedRating(i,j) = sum(Deviation(j,pass_item)+pass_rating)/numel(pass_rating);
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