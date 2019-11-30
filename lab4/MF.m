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

[Precision_1,Recall_1,F1Score_1,TestLoss_1,~,~,~,~,TrainLoss_1,~,~] = MF_Function(train_small_1,20,0.00005,0.2);
[Precision_2,Recall_2,F1Score_2,TestLoss_2,~,~,~,~,TrainLoss_2,~,~] = MF_Function(train_small_2,40,0.00005,0.2);
[Precision_3,Recall_3,F1Score_3,TestLoss_3,~,~,~,~,TrainLoss_3,~,~] = MF_Function(movielens_100k,40,0.00005,0.2);
[Precision_4,Recall_4,F1Score_4,TestLoss_4,~,~,~,~,TrainLoss_4,~,~] = MF_Function(movielens_1m,40,0.00005,0.2);

function [Precision,Recall,F1Score,TestLoss,TrainRating,TestRating,P,Q,TrainLoss,PredictedRating,Indicator] = MF_Function(Data,K,alpha,beta)

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

% ���о���ֽ�
P = rand(nUser,K);                                % �û����Ӿ���P
Q = rand(K,nItem);                                % ��Ʒ���Ӿ���Q
TrainIndicator = (TrainRating~=0);                % ѵ��������Ԫ��ָʾ����
ErrorRating = TrainRating-(P*Q).*TrainIndicator;  % ԭʼ�������ع���������
TrainLoss = sum(sum(ErrorRating.^2));             % ѵ����ʧ��������ʼֵ��
% ͨ���ݶ��½��������û����Ӿ������Ʒ���Ӿ���ֱ����ʧ��������
for t = 1:10000
    % ����P��Q
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
    % ���¼������
    ErrorRating = TrainRating-(P*Q).*TrainIndicator;
    % ���¼�����ʧ����
    TrainLoss = [TrainLoss sum(sum(ErrorRating.^2));];
    % ��ʧ��������ʱֹͣ����
    if TrainLoss(end-1)-TrainLoss(end) <= 0.001
        break;
    end
end

% �����û����Ӿ������Ŀ���Ӿ���ĳ˻�Ԥ������
TestLoss = 0;                         % ��ʧ����
PredictedRating = P*Q;                % Ԥ�����־���
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
        % ����Ԥ������ֵķ�Χ
        if isnan(PredictedRating(i,j)) || isinf(PredictedRating(i,j))
            PredictedRating(i,j) = 0;
        elseif PredictedRating(i,j) > 5
            PredictedRating(i,j) = 5;
        elseif PredictedRating(i,j) < 0
            PredictedRating(i,j) = 0;
        end
        % �����û�û�����ֵ���Ŀ��Ԥ������
        if TrainRating(i,j) == 0
            % ��Ԥ�����ִ���������ֵ�����Ƽ������û�����ָʾ�����ϼ�¼
            if PredictedRating(i,j) >= threshold
                Indicator(i,j) = Indicator(i,j)+2;
            end
            % �����Լ��ϴ��ڸ���Ϊ���������ʧ����������ָʾ�����ϼ�¼
            if TestRating(i,j) ~= 0
                TestLoss = TestLoss+power(TestRating(i,j)-PredictedRating(i,j),2);
                Indicator(i,j) = Indicator(i,j)+1;
            end
        end
    end
end

% ��������ָ��
TestLoss = sqrt(TestLoss/nTest);                       % ������Լ���ʧ����
Precision = sum(sum(Indicator==3))...                  % ����׼ȷ��
    /sum(sum(Indicator==3|Indicator==2));
Recall = sum(sum(Indicator==3))...                     % �����ٻ���
    /sum(sum(Indicator==3|Indicator==1));
F1Score = (2*Precision*Recall)/(Precision+Recall);     % ����F1ֵ

end