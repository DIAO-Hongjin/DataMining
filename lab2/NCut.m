cornell = load('dataset\cornell\cornell.mat');
texas = load('dataset\texas\texas.mat');
washington = load('dataset\washington\washington.mat');
wisconsin = load('dataset\wisconsin\wisconsin.mat');
cora_udir = load('dataset\cora_udir.mat');
polblog = load('dataset\polblog.mat');
polbook = load('dataset\polbook.mat');

data = cornell;
% data = texas ;
% data = washington;
% data = wisconsin;
[nVertex,nFeature] = size(data.F);
% data = polbook;
% [nVertex,~] = size(data.A);

% �õ����ƶȾ���
% W = data.A;                                         % ʹ���ڽӾ��������ƶȾ���
% W = 10./(pdist2(data.F,data.F,'euclidean')+10);     % ʹ��ŷ�����빹�����ƶȾ���ȫ���ӣ�
% W = 200./(pdist2(data.F,data.F,'cityblock')+200);   % ʹ�ý������빹�����ƶȾ���ȫ���ӣ�
% W = 1-pdist2(data.F,data.F,'hamming');              % ʹ�ú������빹�����ƶȾ���ȫ���ӣ�
% W = 1-pdist2(data.F,data.F,'jaccard');              % ʹ��Jaccard���ƶȹ������ƶȾ���ȫ���ӣ�
W = 1-pdist2(data.F,data.F,'cosine');               % ʹ���������ƶȹ������ƶȾ���ȫ���ӣ�
% ȡtop_k���ƶȹ���K����ͼ
top_k = 60;                                           
[max_similarity, index_order] = sort(W,2,'descend');  % �����ƶȽ�������
W = zeros(nVertex);
for i = 1:nVertex                                     % ����K����ͼ��Ϊ���ƶȾ���
    for j = 2:(top_k+1)
        W(i,index_order(i,j)) = max_similarity(i,j);
        W(index_order(i,j),i) = max_similarity(i,j);
    end
end
figure;
plot(graph(W));

% �����׼��������˹����
D = diag(sum(W));
L = D^-0.5*(D-W)*D^-0.5;

% ����������������ֵ��С��������
[V,D1] = eig(L);
[D_sort,index1] = sort(diag(D1));
D_sort = D_sort(index1);
V_sort = V(:,index1);

% ȡǰnС����ֵ��Ӧ�ĵ���������
n = 20;
Y = V_sort(:,1:n);
for i = 1:nVertex
    Y(i,:) = Y(i,:)/norm(Y(i,:));
end

% ʹ��K-means����Ԥ����
k = 40;
% ѡ����Խ�Զ�ĵ���Ϊ����
cur_center = zeros(k,n);
cur_center(1,:) = Y(randi(nVertex),:);
for i = 2:k
    distance = pdist2(Y,cur_center(1:i-1,:));
    [~,index2] = max(min(distance,[],2));
    cur_center(i,:) = Y(index2,:);
end
% �Ż�K-meansĿ�꺯�����о���
label = zeros(nVertex,1);
for t = 1:10000
    % ��ÿ��������
    distance = pdist2(Y,cur_center);
    [~,index3] = min(distance,[],2);
    label = index3;
    % ���¼������
    new_center = zeros(k,n);
    cluster_size = zeros(k,1);
    for i = 1:k
        for j = 1:nVertex
            if label(j) == i
                new_center(i,:) = new_center(i,:)+Y(j,:);
                cluster_size(i) = cluster_size(i)+1;
            end
        end
        new_center(i,:) = new_center(i,:)/cluster_size(i);
    end
    % �ж��Ƿ�����
    if new_center == cur_center
        break;
    else
        cur_center = new_center;
    end
end
% % ����K-means�����Ľ��
% kmeans_label = label;
% temp_graph = zeros(nVertex);
% for i = 1:nVertex
%     for j = (i+1):nVertex
%         if kmeans_label(i) == kmeans_label(j)
%             temp_graph(i,j) = 1;
%             temp_graph(j,i) = 1;
%         end
%     end
% end
% figure;
% plot(graph(temp_graph));

% �ϲ����Եõ����մػ���
nCommunity = 5;            % ָ�����ջ��ֵ�����������
for t = 1:k-nCommunity     % �����������ﵽָ������ǰ���ϲ�����
    ci = 0;
    cj = 0;
    min_ncut = 10000;
    label_type = unique(label);
    % ���Խ�Ŀǰ���������������ϲ�
    for i = 1:numel(label_type)
        for j = (i+1):numel(label_type)
            temp_label = label;
            temp_label(find(temp_label==label_type(j))) = label_type(i);
            temp_ncut = cal_ncut(W,temp_label);    % �ϲ����������¼���NCut
            if(temp_ncut < min_ncut)    % ��¼��ʹNCut��С��һ�κϲ�
                ci = label_type(i);
                cj = label_type(j);
                min_ncut = temp_ncut;
            end
        end
    end
    label(find(label==cj)) = ci;        % ִ��ʹNCut��С��һ�κϲ�
end
% �������ս��
final_graph = zeros(nVertex);
for i = 1:nVertex
    for j = (i+1):nVertex
        if label(i) == label(j)
            final_graph(i,j) = 1;
            final_graph(j,i) = 1;
        end
    end
end
figure;
plot(graph(final_graph));

% ����������
result = ClusteringMeasure(label,data.label)

% ����NCut��ֵ
function ncut = cal_ncut(W,label)
ncut = 0;
label_type = unique(label);
for i = 1:numel(label_type)
    A = find(label==label_type(i));  % �ҵ������ڵĽ��
    C_A = find(label~=label_type(i));% �ҵ�������Ľ��
    cut = sum(sum(W(A,C_A)));        % ����������ıߵ�Ȩ��֮��
    assoc = sum(sum(W(A,:)));        % ���������ڽ�������н��ıߵ�Ȩ��֮��
    ncut = ncut+cut/assoc;           % ����NCut
end
end