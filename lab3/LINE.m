cornell = load('dataset\cornell\cornell.mat');
texas = load('dataset\texas\texas.mat');
washington = load('dataset\washington\washington.mat');
wisconsin = load('dataset\wisconsin\wisconsin.mat');

data = cornell;
% data = texas ;
% data = washington;
% data = wisconsin;

% ��ȡ����ͼ�Ķ�������������ά��
[nVertex,nFeature] = size(data.F);

% ��ȡ����ͼ�����б��Լ��ߵ�����
[source,target] = find(data.A);                 % �õ�ͼ�����бߵ�ʼ����յ�
nEdge = numel(source);                          % �ߵ�����

% ��������
d = 60;
rho_init = 0.025;
K = 5;
T = 10000;
M = 1e8;

% ��ʼ��
U1 = rand(nVertex,d);          % ����һ�����ƶȵ�Ƕ���ʾ
U2 = rand(nVertex,d);          % ���ڶ������ƶȵ�Ƕ���ʾ
U_ = rand(nVertex,d);          % ���ڶ������ƶ�ʱ�ڵ���Ϊ�ھӵı��

% ����ÿ���ڵ�ĳ���(���ڸ�����)
Len = sum(data.A,2).^(3/4);
per_len = sum(Len)/M;

% �Ż�Ŀ�꺯��O1
rho = rho_init;
AVG_ACC1 = [];
for i = 1:T
    % ����ݶ��½������ѡȡһ����
    current_edge = randi(nEdge);
    vi = source(current_edge);
    vj = target(current_edge);
    
    % ������
    vk = zeros(K,1);
    pos = randperm(M,K);
    for j = 1:K                   % ������K����
        pointer = pos(j)*per_len;
        for x = 1:nVertex
            pointer = pointer-Len(x);
            if pointer < 0
                vk(j) = x;
                break;
            end
        end
    end
    
    % ���ݶȼ�����
    delta_uk = zeros(K,d);
    second = 0;
    for k = 1:K
        second = second...
            +U1(vk(k),:)*sigmoid(dot(U1(vk(k),:)',U1(vi,:)));
        delta_uk(k,:) = U1(vi,:)...             % ��uk���ݶ�
            *sigmoid(dot(U1(vk(k),:)',U1(vi,:)));
    end
    delta_ui = -(U1(vj,:)...                    % ��ui���ݶ�
        -U1(vj,:)*sigmoid(dot(U1(vj,:)',U1(vi,:)))-second);
    delta_uj = -U1(vi,:)...                     % ��uj���ݶ�
        *(1-sigmoid(dot(U1(vj)',U1(vi))));
    U1(vk,:) = U1(vk,:)-rho*delta_uk;           % ����u
    U1(vi,:) = U1(vi,:)-rho*delta_ui;           % ����ui
    U1(vj,:) = U1(vj,:)-rho*delta_uj;           % ����uj
    
    % ѧϰ�ʶ�̬��С
    rho = rho_init*(1-i/T);
end

% �Ż�Ŀ�꺯��O2
rho = rho_init;
AVG_ACC2 = [];
for i = 1:T
    % ����ݶ��½������ѡȡһ����
    current_edge = randi(nEdge);
    vi = source(current_edge);
    vj = target(current_edge);
    
    % ������
    vk = zeros(K,1);
    pos = randperm(M,K);
    for j = 1:K                   % ������K����
        pointer = pos(j)*per_len;
        for x = 1:nVertex
            pointer = pointer-Len(x);
            if pointer < 0
                vk(j) = x;
                break;
            end
        end
    end
    
    % ���ݶȼ�����
    delta_uk = zeros(K,d);
    second = 0;
    for k = 1:K
        second = second...
            +U_(vk(k),:)*sigmoid(dot(U_(vk(k),:)',U2(vi,:)));
        delta_uk(k,:) = U2(vi,:)...             % ��uk'���ݶ�
            *sigmoid(dot(U_(vk(k),:)',U2(vi,:)));
    end
    delta_ui = -(U_(vj,:)...                    % ��ui���ݶ�
        -U_(vj,:)*sigmoid(dot(U_(vj,:)',U2(vi,:)))-second);
    delta_uj = -U2(vi,:)...                     % ��uj'���ݶ�
        *(1-sigmoid(dot(U_(vj)',U2(vi))));
    U_(vk,:) = U_(vk,:)-rho*delta_uk;           % ����uk'
    U2(vi,:) = U2(vi,:)-rho*delta_ui;           % ����ui
    U_(vj,:) = U_(vj,:)-rho*delta_uj;           % ����uj'
    
    % ѧϰ�ʶ�̬��С
    rho = rho_init*(1-i/T);
end

% ���ִ�з�������ȡACC��ƽ��ֵ
% ������libsvm���߰�

% % ����һ�����ƶ�
% SUM_ACC1 = 0;
% for i = 1:30
%     nTrain = round(0.8*nVertex);                               % ѵ��������Ϊ80%
%     TrainIndex = randperm(nVertex,nTrain);                     % �����ȡ80%����Ϊѵ����
%     TestIndex = setdiff(1:nVertex,TrainIndex);                 % ʣ��Ϊѵ����
%     Train = U1(TrainIndex,:);                                  % ѵ��������
%     Test = U1(TestIndex,:);                                    % ���Լ�����
%     TrainLabel = data.label(TrainIndex);                       % ���Լ���������ǩ
%     TestLabel = data.label(TestIndex);                         % ���Լ���������ǩ
%     model = svmtrain(TrainLabel,Train,'-t 2 -c 1 -g 0.07');    % ѵ��svmģ��
%     PredictLabel = svmpredict(TestLabel, Test, model);         % ʹ��ѵ���õ�svmģ��Ԥ��
%     ACC1 = classificationACC(TestLabel,PredictLabel);          % ��������Ч��
%     SUM_ACC1 = SUM_ACC1+ACC1;
% end
% AVG_ACC1 = SUM_ACC1/30;
% 
% % ���ڶ������ƶ�
% SUM_ACC2 = 0;
% for i = 1:30
%     nTrain = round(0.8*nVertex);                               % ѵ��������Ϊ80%
%     TrainIndex = randperm(nVertex,nTrain);                     % �����ȡ80%����Ϊѵ����
%     TestIndex = setdiff(1:nVertex,TrainIndex);                 % ʣ��Ϊѵ����
%     Train = U2(TrainIndex,:);                                  % ѵ��������
%     Test = U2(TestIndex,:);                                    % ���Լ�����
%     TrainLabel = data.label(TrainIndex);                       % ���Լ���������ǩ
%     TestLabel = data.label(TestIndex);                         % ���Լ���������ǩ
%     model = svmtrain(TrainLabel,Train,'-t 2 -c 1 -g 0.07');    % ѵ��svmģ��
%     PredictLabel = svmpredict(TestLabel, Test, model);         % ʹ��ѵ���õ�svmģ��Ԥ��
%     ACC2 = classificationACC(TestLabel,PredictLabel);          % ��������Ч��
%     SUM_ACC2 = SUM_ACC2+ACC2;
% end
% AVG_ACC2 = SUM_ACC2/30;
% 
% [AVG_ACC1 AVG_ACC2]

function result = sigmoid(x)
result = 1/(1+exp(-x));
end