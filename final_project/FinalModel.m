% �������ݼ�
% �����Ե�С���ݼ�
cornell = load('dataset\cornell.mat');
texas = load('dataset\texas.mat');
washington = load('dataset\washington.mat');
wisconsin = load('dataset\wisconsin.mat');
% �����ԵĴ����ݼ�
BlogCatalog = load('dataset\BlogCatalog.mat');
Flickr = load('dataset\Flickr.mat');

% cornell���ݼ�
cornell_label = cornell.label;
cornell_representations = SDANE_Function(cornell,'cornell',0.2,3,8,0.1,0.1,0.001,0.01,200,50);
classification_measure1 = classification(cornell_representations,cornell_label);
clustering_measure1 = clustering(cornell_representations,cornell_label,5);

% cornell���ݼ��������ԣ���alpha=1��
% cornell_label = cornell.label;
% cornell_representations = SDANE_Function(cornell,'cornell',1,0,8,0,0.1,0,0.005,0,50);
% classification_measure1 = classification(cornell_representations,cornell_label);
% clustering_measure1 = clustering(cornell_representations,cornell_label,5);

% cornell���ݼ���ȫ���ԣ���alpha=0��
% cornell_label = cornell.label;
% cornell_representations = SDANE_Function(cornell,'cornell',0,3,0,0.1,0.1,0.001,0.01,200,50);
% classification_measure1 = classification(cornell_representations,cornell_label);
% clustering_measure1 = clustering(cornell_representations,cornell_label,5);

% texas���ݼ�
texas_label = texas.label;
texas_representations = SDANE_Function(texas,'texas',0.2,3,8,0.1,0.1,0.001,0.01,200,50);
classification_measure2 = classification(texas_representations,texas_label);
clustering_measure2 = clustering(texas_representations,texas_label,5);

% texas���ݼ��������ԣ���alpha=1��
% texas_label = texas.label;
% texas_representations = SDANE_Function(texas,'texas',1,0,8,0,0.1,0,0.005,0,50);
% classification_measure2 = classification(texas_representations,texas_label);
% clustering_measure2 = clustering(texas_representations,texas_label,5);

% texas���ݼ���ȫ���ԣ���alpha=0��
% texas_label = texas.label;
% texas_representations = SDANE_Function(texas,'texas',0,3,0,0.1,0.1,0.001,0.01,200,50);
% classification_measure2 = classification(texas_representations,texas_label);
% clustering_measure2 = clustering(texas_representations,texas_label,5);

% washington���ݼ�
washington_label = washington.label;
washington_representations = SDANE_Function(washington,'washington',0.2,3,8,0.1,0.1,0.001,0.01,200,50);
classification_measure3 = classification(washington_representations,washington_label);
clustering_measure3 = clustering(washington_representations,washington_label,5);

% washington���ݼ��������ԣ���alpha=1��
% washington_label = washington.label;
% washington_representations = SDANE_Function(washington,'washington',1,0,8,0,0.1,0,0.005,0,50);
% classification_measure3 = classification(washington_representations,washington_label);
% clustering_measure3 = clustering(washington_representations,washington_label,5);

% washington���ݼ���ȫ���ԣ���alpha=0��
% washington_label = washington.label;
% washington_representations = SDANE_Function(washington,'washington',0,3,0,0.1,0.1,0.001,0.01,200,50);
% classification_measure3 = classification(washington_representations,washington_label);
% clustering_measure3 = clustering(washington_representations,washington_label,5);

% wisconsin���ݼ�
wisconsin_label = wisconsin.label;
wisconsin_representations = SDANE_Function(wisconsin,'wisconsin',0.2,3,8,0.1,0.1,0.001,0.01,200,50);
classification_measure4 = classification(wisconsin_representations,wisconsin_label);
clustering_measure4 = clustering(wisconsin_representations,wisconsin_label,5);

% wisconsin���ݼ��������ԣ���alpha=1��
% wisconsin_label = wisconsin.label;
% wisconsin_representations = SDANE_Function(wisconsin,'wisconsin',1,0,8,0,0.1,0,0.005,0,50);
% classification_measure4 = classification(wisconsin_representations,wisconsin_label);
% clustering_measure4 = clustering(wisconsin_representations,wisconsin_label,5);

% wisconsin���ݼ���ȫ���ԣ���alpha=0��
% wisconsin_label = wisconsin.label;
% wisconsin_representations = SDANE_Function(wisconsin,'wisconsin',0,3,0,0.1,0.1,0.001,0.01,200,50);
% classification_measure4 = classification(wisconsin_representations,wisconsin_label);
% clustering_measure4 = clustering(wisconsin_representations,wisconsin_label,5);

% BlogCatalog���ݼ�
% BlogCatalog_label = BlogCatalog.Label;
% BlogCatalog_representations = SDANE_Function(BlogCatalog,'BlogCatalog',0.4,15,10,0.1,0.1,0.00005,0.005,500,100);
% classification_measure = classification(BlogCatalog_representations,BlogCatalog_label);
% clustering_measure = clustering(BlogCatalog_representations,BlogCatalog_label,6);

% BlogCatalog���ݼ��������ԣ���alpha=1) 
% BlogCatalog_label = BlogCatalog.Label;
% BlogCatalog_representations = SDANE_Function(BlogCatalog,'BlogCatalog',1,0,10,0.1,0.1,0,0.0005,0,100);
% classification_measure = classification(BlogCatalog_representations,BlogCatalog_label);
% clustering_measure = clustering(BlogCatalog_representations,BlogCatalog_label,6);

% BlogCatalog���ݼ���ȫ���ԣ���alpha=0��
% BlogCatalog_label = BlogCatalog.Label;
% BlogCatalog_representations = SDANE_Function(BlogCatalog,'BlogCatalog',0,15,0,0.1,0.1,0.00005,0.005,500,100);
% classification_measure = classification(BlogCatalog_representations,BlogCatalog_label);
% clustering_measure = clustering(BlogCatalog_representations,BlogCatalog_label,6);

% Flickr���ݼ�
% Flickr_label = Flickr.Label;
% Flickr_representations = SDANE_Function(Flickr,'Flickr',0.4,30,15,0.1,0.1,0.0001,0.00001,500,100);
% classification_measure = classification(Flickr_representations,Flickr_label);
% clustering_measure = clustering(Flickr_representations,Flickr_label,9);

% Flickr���ݼ��������ԣ���alpha=1)
% Flickr_label = Flickr.Label;
% Flickr_representations = SDANE_Function(Flickr,'Flickr',1,0,15,0,0.1,0,0.000005,0,100);
% classification_measure = classification(Flickr_representations,Flickr_label);
% clustering_measure = clustering(Flickr_representations,Flickr_label,9);

% Flickr���ݼ���ȫ���ԣ���alpha=0��
% Flickr_label = Flickr.Label;
% Flickr_representations = SDANE_Function(Flickr,'Flickr',0,30,0,0.1,0.1,0.0001,0.00001,500,100);
% classification_measure = classification(Flickr_representations,Flickr_label);
% clustering_measure = clustering(Flickr_representations,Flickr_label,9);

% SDANEѧϰ�ڵ��Ƕ���ʾ
% ������������ݼ������ݼ����ơ�������alpha���ڵ����Եķ���Ԫ��Ȩ�ء��ڵ����˵ķ���Ԫ��Ȩ�ء�
% �ڵ�����������ȡ�����򻯲������ڵ��ʾѧϰ�����򻯲������ڵ�����������ȡ�Ĳ������ڵ��ʾ�Ĳ�����
% �ڵ����Խ�ά���ά�ȡ��ڵ��ʾ��ά��
function node_representations = SDANE_Function(data,name,alpha,beta1,beta2,lambda1,lambda2,eta1,eta2,K1,K2)

% �����������˽ṹ���ڵ����Ժͽڵ��ǩ
if strcmp(name,'BlogCatalog') || strcmp(name,'Flickr')
    network = data.Network;
    attribute = data.Attributes;
else
    network = data.A;
    attribute = data.F;
end

% ����Ƕ��ģ��
if alpha == 1
    % ͨ��SDAE�õ������нڵ�ı�ʾ
    node_representations = SDAE_Function(network,beta2,lambda2,eta2,K2);
else
    % ͨ��SDAE�õ���ά��Ľڵ�����
    attribute_representations = SDAE_Function(attribute,beta1,lambda1,eta1,K1);
    % ͨ��aSDAE�õ������нڵ�ı�ʾ
    node_representations = aSDAE_Function(attribute_representations,network,alpha,beta2,lambda2,eta2,K2);
end

end

% SDAE��ȡ�ڵ����Ե�����������ά��
% ����������ڵ����ԡ�����Ԫ��Ȩ�ء�����ϵ������������ά��ά��
function attribute_representations = SDAE_Function(attribute,beta,lambda,eta,K)

% % Ŀ�꺯��
% L1 = [];

% ������
n_sample = size(attribute,1);

% ÿ���ά��
input_size = size(attribute,2);        % ������ά��
hidden_size = K;                       % ���ز��ά�ȣ����ڵ����Խ�ά���ά�ȣ�
output_size = input_size;              % ������ά��

% ϡ���Լ�Ȩ����
B1 = 1*(attribute==0)+beta*(attribute~=0);

% ��ʼ��Ȩ�ؾ����ƫ������
r = sqrt(6)/sqrt(hidden_size+output_size+1);
U1 = rand(hidden_size,input_size)*2*r-r;     % ����㵽���ز�֮���Ȩ��
U2 = rand(output_size,hidden_size)*2*r-r;    % ���ز㵽�����֮���Ȩ��
b11 = zeros(hidden_size,1);                  % ���ز��ƫ��
b12 = zeros(output_size,1);                  % ������ƫ��

% % Ŀ�꺯���ĳ�ʼֵ
% L1 = [L1 objective1(attribute,beta,lambda,U1,U2,b11,b12)];

% ���ѵ��
% ��һ�㣨����ѵ����
corrupt_attr = attribute.*(rand(n_sample,input_size)>0.3);    % �����������������
% for i = 1:5000
for i = 1:10000
    % ����ݶ��½���
    index = randi(n_sample);
    % ǰ�򴫲�
    % ���ز�
    input11 = (U1*corrupt_attr(index,:)'+b11)';     % ����
    output11 = tanh(input11);                       % ���
    % �����
    input12 = (U2*output11'+b12)';                  % ����
    output12 = input12;                             % ���
    % ���򴫵�
    % �����
    delta2 = (output12-attribute(index,:)).*B1(index,:);    % �����ļ�Ȩ���
    U2 = U2-eta*(delta2'*output11+lambda*U2);               % ����Ȩ��
    b12 = b12-eta*delta2';                                  % ����ƫ��
    % ���ز�
    % delta1 = sum(U2.*(delta2'*(1-output11.^2)));            % �������
    delta1 = ((U2'*delta2').*(1-output11.^2)')';            % �������
    U1 = U1-eta*(delta1'*corrupt_attr(index,:)+lambda*U1);  % ����Ȩ��
    b11 = b11-eta*delta1';                                  % ����ƫ��
    
%     % Ŀ�꺯���ı仯���
%     if mod(i,50) == 0
%         L1 = [L1 objective1(attribute,beta,lambda,U1,U2,b11,b12)];
%     end
end

% % Ŀ�꺯��L1�ı仯���
% figure;
% plot(1:numel(L1),L1);

% �ڶ������ز�������Ϊ��ά��Ľڵ�����
attribute_representations = tanh(U1*attribute'+repmat(b11,1,n_sample))';

end

% Ŀ�꺯��L1
function loss = objective1(attribute,beta,lambda,U1,U2,b11,b12)

% ������
n_sample = size(attribute,1);

% ϡ���Լ�Ȩ����
B1 = 1*(attribute==0)+beta*(attribute~=0);

% ���ز�
input1 = (U1*attribute'+repmat(b11,1,n_sample))';    % ����
output1 = tanh(input1);                              % ���
% �����
input2 = (U2*output1'+repmat(b12,1,n_sample))';      % ����
output2 = input2;                                    % ���

% ����L1
loss = 0.5*norm((output2-attribute).*B1,'fro')^2+0.5*lambda*(norm(U1,'fro')^2+norm(U2,'fro')^2);

end

% aSDAEѧϰ����Ľڵ��ʾ
% ����������ڵ����ԡ�������alpha������Ԫ��Ȩ�ء�����ϵ����������Ƕ���ʾ��ά��
function node_representations = aSDAE_Function(attribute_representations,network,alpha,beta,lambda,eta,K)

% % Ŀ�꺯��
% L2 = [];

% ������
n_sample = size(network,1);
% ����ά��
n_attr_dim = size(attribute_representations,2);

% ÿ���ά��
input_size = size(network,2);           % ������ά��
hidden_size = K;                        % ���ز��ά�ȣ����ڵ�Ƕ���ʾ��ά�ȣ�
outputX_size = input_size;              % ������ά��
outputZ_size = n_attr_dim;

% ϡ���Լ�Ȩ����
B2 = 1*(network==0)+beta*(network~=0);

% ��ʼ��Ȩ�ؾ����ƫ������
r1 = sqrt(6)/sqrt(hidden_size+outputX_size+1);
r2 = sqrt(6)/sqrt(hidden_size+outputZ_size+1);
W1 = rand(hidden_size,input_size)*2*r1-r1;      % ����㵽���ز�֮���Ȩ��
W2 = rand(outputX_size,hidden_size)*2*r1-r1;    % ���ز㵽����㣨�ع��ڽӾ���֮���Ȩ��
V1 = rand(hidden_size,n_attr_dim)*2*r2-r2;      % �ڵ����Ե����ز�֮���Ȩ��
V2 = rand(outputZ_size,hidden_size)*2*r2-r2;    % ���ز㵽����㣨�ع�����������֮���Ȩ��
b21 = zeros(hidden_size,1);                     % ���ز��ƫ��
b22X = zeros(outputX_size,1);                   % ����㣨�ع��ڽӾ��󣩵�ƫ��
b22Z = zeros(outputZ_size,1);                   % ����㣨�ع�������������ƫ��

% % Ŀ�꺯���ĳ�ʼֵ
% L2 = [L2 objective2(attribute_representations,network,alpha,beta,lambda,V1,V2,W1,W2,b21,b22X,b22Z)];

% ����ѵ��
corrupt_network = network.*(rand(n_sample,input_size)>0.3);    % �����������������
% for i = 1:5000
for i = 1:10000
    % ����ݶ��½���
    index = randi(n_sample);
    % ǰ�򴫲�
    % ���ز�
    input1 = (W1*corrupt_network(index,:)'+V1*attribute_representations(index,:)'+b21)';    % ����
    output1 = tanh(input1);                                                                 % ���
    % �����
    input2X = (W2*output1'+b22X)';                                                          % ����
    input2Z = (V2*output1'+b22Z)';
    output2X = sigmoid(input2X);                                                            % ���
    output2Z = input2Z;  
   
    % ���򴫵�
    % �����
    delta2X = alpha*((output2X-network(index,:)).*B2(index,:)).*(output2X.*(1-output2X));    % �����ļ�Ȩ���
    delta2Z = (1-alpha)*(output2Z-attribute_representations(index,:));
    W2 = W2-eta*(delta2X'*output1+lambda*W2);                                                % ����Ȩ��
    V2 = V2-eta*(delta2Z'*output1+lambda*V2);
    b22X = b22X-eta*delta2X';                                                                % ����ƫ��
    b22Z = b22Z-eta*delta2Z';
    % ���ز�
%     delta1X = sum(W2.*(delta2X'*(1-output1.^2)));                                            % �������
%     delta1Z = sum(V2.*(delta2Z'*(1-output1.^2)));
    delta1X = ((W2'*delta2X').*(1-output1.^2)')';                                            % �������
    delta1Z = ((V2'*delta2Z').*(1-output1.^2)')';
    W1 = W1-eta*(delta1X'*corrupt_network(index,:)+lambda*W1);                               % ����Ȩ��
    V1 = V1-eta*(delta1Z'*attribute_representations(index,:)+lambda*V1);
    b21 = b21-eta*(delta1X'+delta1Z');                                                       % ����ƫ��
   
%     % Ŀ�꺯���ı仯���
%     if mod(i,50) == 0
%         L2 = [L2 objective2(attribute_representations,network,alpha,beta,lambda,V1,V2,W1,W2,b21,b22X,b22Z)];
%     end
end

% % Ŀ�꺯��L2�ı仯���
% figure;
% plot(1:numel(L2),L2);

% �ڶ������ز�������Ϊ��ά��Ľڵ�����
node_representations = tanh(W1*network'+V1*attribute_representations'+repmat(b21,1,n_sample))';

end

% Ŀ�꺯��L2
function loss = objective2(attribute_representations,network,alpha,beta,lambda,V1,V2,W1,W2,b21,b22X,b22Z)

% ������
n_sample = size(network,1);

% ϡ���Լ�Ȩ����
B2 = 1*(network==0)+beta*(network~=0);

% ���ز�
input1 = (W1*network'+V1*attribute_representations'+repmat(b21,1,n_sample))';    % ����
output1 = tanh(input1);                                                          % ���
% �����
input2X = (W2*output1'+repmat(b22X,1,n_sample))';                                % ����
input2Z = (V2*output1'+repmat(b22Z,1,n_sample))';
output2X = sigmoid(input2X);                                                     % ���
output2Z = input2Z;                                    

% ����L2
loss = 0.5*alpha*norm((output2X-network).*B2,'fro')^2 ...
    +0.5*(1-alpha)*norm((output2Z-attribute_representations),'fro')^2 ...
    +0.5*lambda*(norm(V1,'fro')^2+norm(V2,'fro')^2+norm(W1,'fro')^2+norm(W2,'fro')^2);

end

% ��������
function classification_measure = classification(node_representations,label)

% ������
n_sample = size(node_representations,1); 

classification_measure = 0;
for i = 1:5
    % ���ֲ��Լ���ѵ����
    test_index = i:5:n_sample;                          % ���Լ�����
    train_index = setdiff(1:n_sample,test_index);       % ѵ��������
    test_set = node_representations(test_index,:);      % ���Լ�
    train_set = node_representations(train_index,:);    % ѵ����
    test_label = label(test_index,:);                   % ���Լ�������ǩ
    train_label = label(train_index,:);                 % ѵ����������ǩ
    
    % ����libsvm��������з���
    model = svmtrain(train_label,train_set,'-t 2 -c 1 -g 0.07');                                      % ѵ��svmģ��
    predict_label = svmpredict(test_label,test_set,model);                                            % ʹ��ѵ���õ�svmģ��Ԥ��
    classification_measure = classification_measure + classificationACC(test_label,predict_label);    % ��������Ч��
end

classification_measure = classification_measure/5;      % ȡƽ��ֵ��Ϊ���ս��

end

% ��������
function clustering_measure = clustering(node_representations,label,cluster_num)

clustering_measure = zeros(1,3);
for i = 1:5
    % �ڵ��Ƕ���ʾִ��K-Means����
    cluster_id = kmeans(node_representations,cluster_num);
    % ����������
    clustering_measure = clustering_measure+ClusteringMeasure(cluster_id,label);
end

clustering_measure = clustering_measure/5;    % ȡƽ��ֵ��Ϊ���ս��

end

% �����֮һ��sigmoid����
function Y = sigmoid(X)
Y = sigmf(X,[1 0]);
end