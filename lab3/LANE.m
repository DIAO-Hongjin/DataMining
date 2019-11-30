cornell = load('dataset\cornell\cornell.mat');
texas = load('dataset\texas\texas.mat');
washington = load('dataset\washington\washington.mat');
wisconsin = load('dataset\wisconsin\wisconsin.mat');

% data = cornell;
% data = texas ;
% data = washington;
data = wisconsin;

% ��ȡ����ͼ�Ķ�������������ά��
[nVertex,nFeature] = size(data.F);

% ��������
d = 100;
alpha1 = 2;
alpha2 = 15;
epsilon = 0.1;
delta1 = 0.5;
delta2 = 1.5;

% ����ͼ�б�ǩ���������
nCommunity = 5;

% ����ѵ�����Ͳ��Լ�
nTrain = round(0.8*nVertex);                       % ѵ��������Ϊ80%
TrainIndex = randperm(nVertex,nTrain);             % �����ȡ80%����Ϊѵ����
TestIndex = setdiff(1:nVertex,TrainIndex);         % ʣ��Ϊѵ����
G_Train = data.A(TrainIndex,TrainIndex);           % ѵ�������ڽӾ���
A_Train = data.F(TrainIndex,:);                    % ѵ�����Ľڵ�����
Y_Train = zeros(nTrain,nCommunity);                % ѵ�����ı�ǩ����
for i = 1:nTrain
    Y_Train(i,data.label(TrainIndex(i))) = 1;
end
G_Test = data.A(TestIndex,TestIndex);              % ���Լ����ڽӾ���
A_Test = data.F(TestIndex,:);                      % ���Լ��Ľڵ�����
Y_Test = zeros(nVertex-nTrain,nCommunity);         % ���Լ��ı�ǩ����
for i = 1:(nVertex-nTrain)
    Y_Test(i,data.label(TestIndex(i))) = 1;
end

% ����ͼ�����˽ṹ�����ƶȾ��󣺻��ڶ������ƶ�
S_G = zeros(nTrain); 
for i = 1:nTrain     
    for j = 1:nTrain 
        % ���һ�Խ����������һ�������㣬��������ƶ�Ϊ0
        % ������Խ��Ķ������ƶ�Ϊ�������ƶ�
        if norm(G_Train(i,:)) == 0 || norm(G_Train(j,:)) == 0
            S_G(i,j) = 0;
        else
            S_G(i,j) = dot(G_Train(i,:),G_Train(j,:))...
                /(norm(G_Train(i,:))*norm(G_Train(j,:)));
        end
    end
end

% ����ͼ�Ľڵ����Ե����ƶȾ���:�����������ƶ�
S_A = 1-pdist2(A_Train,A_Train,'cosine'); 

% ����ͼ�Ľڵ��ǩ�����ƶȾ��󣺻����������ƶ�
S_Y = 1-pdist2(Y_Train*Y_Train',Y_Train*Y_Train','cosine'); 

% �ֱ�������˽ṹ���ڵ����Ժͽڵ��ǩ��������˹����
D_G = diag(sum(S_G,2));                    % �������˽ṹ�����ƶȾ���ĶȾ���
L_G = (D_G^-0.5)*S_G*(D_G^-0.5);           % �������˽ṹ��������˹����
L_G(isnan(L_G)) = 0;                       % ������ܳ��ֵ�NAN
L_G = 0.5*(L_G+L_G');                      % ����Ϊ�Գƾ���
D_A = diag(sum(S_A,2));                    % ����ڵ����Ե����ƶȾ���ĶȾ���
L_A = (D_A^-0.5)*S_A*(D_A^-0.5);           % ����ڵ����Ե�������˹����
L_A(isnan(L_A)) = 0;                       % ������ܳ��ֵ�NAN
L_A = 0.5*(L_A+L_A');                      % ����Ϊ�Գƾ���
D_Y = diag(sum(S_Y,2));                    % ����ڵ��ǩ�����ƶȾ���ĶȾ���
L_Y = (D_Y^-0.5)*S_Y*(D_Y^-0.5);           % ����ڵ��ǩ��������˹����
L_Y(isnan(L_Y)) = 0;                       % ������ܳ��ֵ�NAN
L_Y = 0.5*(L_Y+L_Y');                      % ����Ϊ�Գƾ���

% ��ʼ����ʾ����
U_G = zeros(nTrain,d);                     % ���˽ṹ��Ǳ�ڱ�ʾ
U_A = zeros(nTrain,d);                     % �ڵ����Ե�Ǳ�ڱ�ʾ
U_Y = zeros(nTrain,d);                     % �ڵ��ǩ��Ǳ�ڱ�ʾ
H_Train = zeros(nTrain,d);                 % ѵ��������Ƕ����

% �Ż�Ŀ�꺯��
t = 1;                                     % ��¼��������
objective_0 = trace(U_G'*L_G*U_G)...       % ���������Ŀ�꺯��
    +alpha1*trace(U_A'*L_A*U_A)...
    +alpha1*trace(U_A'*(U_G*U_G')*U_A)...
    +alpha2*trace(U_Y'*(L_Y+U_G*U_G')*U_Y)...
    +trace(U_G'*(H_Train*H_Train')*U_G)...
    +trace(U_A'*(H_Train*H_Train')*U_A)...
    +trace(U_Y'*(H_Train*H_Train')*U_Y);
objective = [objective_0];
while true                                 % �������¸�������
    % ����U_G
    M1 = L_G+alpha1*(U_A*U_A')+alpha2*(U_Y*U_Y')+(H_Train*H_Train');
    [U_G,~] = eigs(M1,d);
    
    % ����U_A
    M2 = alpha1*L_A+alpha1*(U_G*U_G')+(H_Train*H_Train');
    [U_A,~] = eigs(M2,d);
    
    % ����U_Y
    M3 = alpha2*L_Y+alpha2*(U_G*U_G')+(H_Train*H_Train');
    [U_Y,~] = eigs(M3,d);
    
    % ����H
    M4 = (U_G*U_G')+(U_A*U_A')+(U_Y*U_Y');
    [H_Train,~] = eigs(M4,d);
    
    % ����������һ
    t = t+1;
    
    % ����Ŀ�꺯��
    objective_t = trace(U_G'*L_G*U_G)...       
        +alpha1*trace(U_A'*L_A*U_A)...
        +alpha1*trace(U_A'*(U_G*U_G')*U_A)...
        +alpha2*trace(U_Y'*(L_Y+U_G*U_G')*U_Y)...
        +trace(U_G'*(H_Train*H_Train')*U_G)...
        +trace(U_A'*(H_Train*H_Train')*U_A)...
        +trace(U_Y'*(H_Train*H_Train')*U_Y);
    objective = [objective objective_t];
    
    % Ŀ�꺯������ʱֹͣ���� 
    if objective(t)-objective(t-1) < epsilon
        break;
    end
end

% �õ����Լ�������Ƕ���ʾ
G1 = data.A(TrainIndex,:);                      % ѵ���������нڵ���ڽӾ���
G2 = data.A(TestIndex,:);                       % ���Լ������нڵ���ڽӾ���
H_Test = delta1*(G2*pinv(pinv(H_Train)*G1))...  % ���Լ���Ƕ���ʾ
    +delta2*(A_Test*pinv(pinv(H_Train)*A_Train));

% ������libsvm���߰�ִ�з�������
% TrainLabel = data.label(TrainIndex);                       % ���Լ���������ǩ
% TestLabel = data.label(TestIndex);                         % ���Լ���������ǩ
% model = svmtrain(TrainLabel,H_Train,'-t 2 -c 1 -g 0.07');  % ѵ��svmģ��
% PredictLabel = svmpredict(TestLabel, H_Test, model);       % ʹ��ѵ���õ�svmģ��Ԥ��
% ACC = classificationACC(TestLabel,PredictLabel)            % ��������Ч��