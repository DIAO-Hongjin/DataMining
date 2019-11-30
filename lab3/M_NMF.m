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
nDimension = 100;     % ��ά���ά��
nCommunity = 5;      % ����������
alpha = 0.1;
beta = 0.5;
lambda = 1e9;

% ��ȡͼ���ڽӾ���
A = data.A; 

% ����һ�׺Ͷ������ƶȾ���S��S=S1+5*S2
S1 = A;                                                 % ����һ�����ƶ�
S2 = 1-pdist2(S1,S1,'cosine');                          % ����������ƶ�
S = S1+5*S2;                                            % �������S

% �������B1��B1(i,j)=(k_i*k_j)/(2*e)
e = sum(sum(A))/2;                                      % ͼ�бߵ�����
k = sum(A,2);                                           % ͼ�и����Ķ�
B1 = (repmat(k,1,nVertex).*repmat(k',nVertex,1))/(2*e); % �������B1

% ��ʼ��
M = rand(nVertex,nDimension);                           % ��ʼ��������
U = rand(nVertex,nDimension);                           % ��ʼ������ʾ����
C = rand(nCommunity,nDimension);                        % ��ʼ��������ʾ����
H = rand(nVertex,nCommunity);                           % ��ʼ������ָʾ����

% �Ż�Ŀ�꺯��
t = 1;                                                  % ��¼��������
threshold = 1e2;
objective_0 = norm(S-M*U','fro')^2 ...                  % ���������Ŀ�꺯��
    +alpha*norm(H-U*C','fro')^2 ...
    -beta*trace(H'*(A-B1)*H)...
    +lambda*norm(H'*H-eye(nCommunity))^2;
objective = [objective_0];
while true
   % ���¾���M
   M = M.*((S*U)./(M*(U'*U)));
   
   % ���¾���U
   U = U.*((S'*M+alpha*H*C)./(U*((M'*M)+alpha*(C'*C))));
   
   % ���¾���C
   C = C.*((H'*U)./(C*(U'*U)));
   
   % ���¾���H
   delta = (2*beta*(B1*H)).*(2*beta*(B1*H))+16*lambda*(H*(H'*H))...
       .*(2*beta*A*H+2*alpha*U*C'+(4*lambda-2*alpha)*H);
   H = H.*sqrt((-2*beta*B1*H+sqrt(delta))./(8*lambda*H*(H'*H)));
   
   % ����������һ
   t = t+1;
   
   % ����Ŀ�꺯����ֵ
   objective_i = norm(S-M*U','fro')^2+alpha*norm(H-U*C','fro')^2 ...
       -beta*trace(H'*(A-B1)*H)+lambda*norm(H'*H-eye(nCommunity))^2;
   objective = [objective objective_i];
   
   % Ŀ�꺯������ʱֹͣ����
   if objective(t-1)-objective(t) < threshold
       break;
   end
   
   % ����NANʱ����ѭ��
   if isnan(objective(t))
       break;
   end
end

% ���ִ�з�������ȡACC��ƽ��ֵ
% ������libsvm���߰�
% SUM_ACC = 0;
% for i = 1:30
%     nTrain = round(0.8*nVertex);                               % ѵ��������Ϊ80%
%     TrainIndex = randperm(nVertex,nTrain);                     % �����ȡ80%����Ϊѵ����
%     TestIndex = setdiff(1:nVertex,TrainIndex);                 % ʣ��Ϊѵ����
%     Train = U(TrainIndex,:);                                   % ѵ��������
%     Test = U(TestIndex,:);                                     % ���Լ�����
%     TrainLabel = data.label(TrainIndex);                       % ���Լ���������ǩ
%     TestLabel = data.label(TestIndex);                         % ���Լ���������ǩ
%     model = svmtrain(TrainLabel,Train,'-t 2 -c 1 -g 0.07');    % ѵ��svmģ��
%     PredictLabel = svmpredict(TestLabel, Test, model);         % ʹ��ѵ���õ�svmģ��Ԥ��
%     ACC = classificationACC(TestLabel,PredictLabel);           % ��������Ч��
%     SUM_ACC = SUM_ACC+ACC;
% end
% AVG_ACC = SUM_ACC/30