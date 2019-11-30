cornell = load('dataset\cornell\cornell.mat');
texas = load('dataset\texas\texas.mat');
washington = load('dataset\washington\washington.mat');
wisconsin = load('dataset\wisconsin\wisconsin.mat');

% data = cornell;
% data = texas ;
% data = washington;
data = wisconsin;
[nVertex,nFeature] = size(data.F);
% data = polbook;
% [nVertex,~] = size(data.A);

% �õ����ƶȾ���
A = data.A;                                         % ʹ���ڽӾ��������ƶȾ���
% A = 10./(pdist2(data.F,data.F,'euclidean')+10);     % ʹ��ŷ�����빹�����ƶȾ���ȫ���ӣ�
% A = 200./(pdist2(data.F,data.F,'cityblock')+200);   % ʹ�ý������빹�����ƶȾ���ȫ���ӣ�
% A = 1-pdist2(data.F,data.F,'hamming');              % ʹ�ú������빹�����ƶȾ���ȫ���ӣ�
% A = 1-pdist2(data.F,data.F,'jaccard');              % ʹ��Jaccard���ƶȹ������ƶȾ���ȫ���ӣ�
% A = 1-pdist2(data.F,data.F,'cosine');               % ʹ���������ƶȹ������ƶȾ���ȫ���ӣ�
% A = A.*data.A;                                      % ��ԭ�еı߸������ƶȸ�ֵ
% top_k = 20;                                           % ȡtop_k���ƶȹ���K����ͼ
% [max_similarity, index_order] = sort(A,2,'descend');  % �����ƶȽ�������
% A = zeros(nVertex);
% for i = 1:nVertex                                     % ����K����ͼ��Ϊ���ƶȾ���
%     for j = 2:(top_k+1)
%         A(i,index_order(i,j)) = max_similarity(i,j);
%         A(index_order(i,j),i) = max_similarity(i,j);
%     end
% end
% figure;
% plot(graph(A));

% ��ά������ά�ȣ�������������
k = 5;

% ��ʼ������ָʾ����ͻ�����
U = rand(nVertex,k);     % ������
V = rand(nVertex,k);     % ����ָʾ����

% ��������ָʾ����ͻ�����
max_iterator = 45;
for i = 1:max_iterator
    U = U.*(A*V)./(U*(V'*V));  % ���»�����
    V = V.*(A'*U)./(V*(U'*U)); % ��������ָʾ����
end

% ��������ָʾ����ȷ�����
[~,label] = max(V,[],2);

% �������ս��
% final_graph = zeros(nVertex);
% for i = 1:nVertex
%     for j = (i+1):nVertex
%         if label(i) == label(j)
%             final_graph(i,j) = 1;
%             final_graph(j,i) = 1;
%         end
%     end
% end
% figure;
% plot(graph(final_graph));

% ����������
result = ClusteringMeasure(label,data.label)