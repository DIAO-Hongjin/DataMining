cornell = load('dataset\cornell\cornell.mat');
texas = load('dataset\texas\texas.mat');
washington = load('dataset\washington\washington.mat');
wisconsin = load('dataset\wisconsin\wisconsin.mat');

% data = cornell;
% data = texas ;
% data = washington;
data = wisconsin;
[nVertex,nFeature] = size(data.F);

% �õ����ƶȾ���
A = data.A;                                         % ʹ���ڽӾ��������ƶȾ���
% A = 10./(pdist2(data.F,data.F,'euclidean')+10);     % ʹ��ŷ�����빹�����ƶȾ���ȫ���ӣ�
% A = 200./(pdist2(data.F,data.F,'cityblock')+200);   % ʹ�ý������빹�����ƶȾ���ȫ���ӣ�
% A = 1-pdist2(data.F,data.F,'hamming');              % ʹ�ú������빹�����ƶȾ���ȫ���ӣ�
% A = 1-pdist2(data.F,data.F,'jaccard');              % ʹ��Jaccard���ƶȹ������ƶȾ���ȫ���ӣ�
% A = 1-pdist2(data.F,data.F,'cosine');               % ʹ���������ƶȹ������ƶȾ���ȫ���ӣ�
% A = A.*data.A;                                      % ��ԭ�еı߸������ƶȸ�ֵ
% ȡtop_k���ƶȹ���K����ͼ
% top_k = 10;                                           
% [max_similarity, index_order] = sort(A,2,'descend');  % �����ƶȽ�������
% A = zeros(nVertex);
% for i = 1:nVertex                                     % ����K����ͼ��Ϊ���ƶȾ���
%     for j = 2:(top_k+1)
%         A(i,index_order(i,j)) = max_similarity(i,j);
%         A(index_order(i,j),i) = max_similarity(i,j);
%     end
% end
figure;
plot(graph(A));

% �ϲ����
cur_label1 = 1:nVertex; % ÿ������ʼ��Ϊ��ͬ������
for t = 1:10000
    new_label1 = cur_label1;
    for i = 1:nVertex              % �����ڵ�
        neighbor = find(A(i,:));
        max_gain = -10000;
        max_index = 0;
        for j = 1:numel(neighbor)  % ���Խ����i���뵽���ھ����ڵ�������
            temp_label = new_label1;
            temp_label(i) = temp_label(neighbor(j));
            gain = cal_modularity_gain(A,temp_label,i);
            if(gain > max_gain)
                max_gain = gain;
                max_index = neighbor(j);
            end
        end
        if max_gain > 0            %ѡ����ʹ���������ھӽ��кϲ�
            new_label1(i) = new_label1(max_index);
        end
    end
    if new_label1 == cur_label1    % ���������������ٸı�ʱֹͣ��һ����
        break;
    else
        cur_label1 = new_label1;
    end
end

% ��ͼ����ѹ��
label_type = unique(cur_label1);
nCommunity = numel(label_type);
% ��¼�����еĽ��
community_vertex = zeros(nCommunity,nVertex);  
for i = 1:nCommunity
    actual_vertex = find(cur_label1==label_type(i));
    for j = 1:numel(actual_vertex)
        community_vertex(i,j) = actual_vertex(j);
    end
end
% �����µ�����
compression = zeros(nCommunity,nCommunity);    
for i = 1:nCommunity
    community1 = community_vertex(i,find(community_vertex(i,:)));
    % �����ڽ��֮��ıߵ�Ȩ��ת��Ϊ�½��Ļ���Ȩ��
    compression(i,i) = (sum(sum(A(community1,community1)))...
        +sum(diag(A(community1,community1))))/2;
    % ������ıߵ�Ȩ��ת��Ϊ�½��֮��ıߵ�Ȩ��
    for j = (i+1):nCommunity
        community2 = community_vertex(j,find(community_vertex(j,:)));
        compression(i,j) = sum(sum(A(community1,community2)));
        compression(j,i) = sum(sum(A(community1,community2)));
    end
end
figure;
plot(graph(compression));

% �����ϲ�ѹ�����ͼ
cur_label2 = label_type;           % �����ı�ǩ
final_label = cur_label1;          % ���ı�ǩ
for t = 1:10000
    new_label2 = cur_label2;
    for i = 1:nCommunity           % ��������
        neighbor = find(compression(i,:));
        max_gain = -10000;
        max_index = 0;
        for j = 1:numel(neighbor)  % ���Ժϲ��������ڵ�����
            temp_label = new_label2;
            temp_label(i) = temp_label(neighbor(j));
            gain = cal_modularity_gain(compression,temp_label,i);
            if(gain > max_gain)
                max_gain = gain;
                max_index = neighbor(j);
            end
        end
        if max_gain > 0            %ѡ����ʹ���������ھӽ��кϲ�
            new_label2(i) = new_label2(max_index);
            final_label(community_vertex(i,find(community_vertex(i,:))))...
                = new_label2(max_index);
        end
    end
    if new_label2 == cur_label2    % ���������������ٸı�ʱֹͣ��һ����
        break;
    else
        cur_label2 = new_label2;
    end
end

% �������ս��
final_graph = zeros(nVertex);
for i = 1:nVertex
    for j = (i+1):nVertex
        if final_label(i) == final_label(j) && A(i,j) ~= 0
            final_graph(i,j) = 1;
            final_graph(j,i) = 1;
        end
    end
end
figure;
plot(graph(final_graph));

% ����������
result = ClusteringMeasure(final_label,data.label)

% ����ģ�������
function gain = cal_modularity_gain(A,label,i)
vertex_in = find(label==label(i));            % �ҳ����i��������C�е����н��
k_i = sum(A(i,:));                            % ��������i���������бߵĺ�
k_i_in = sum(A(i,vertex_in));                 % ������i������C�ڲ����ıߵ�Ȩ��֮��
sum_tot = sum(sum(A(vertex_in,:)))-...        % ����C��ȫ�����ıߵ�Ȩ��֮��
    (sum(sum(A(vertex_in,vertex_in)))-sum(diag(A(vertex_in,vertex_in))))/2;
m = (sum(sum(A))+sum(diag(A)))/2;             % ������������ıߵ�Ȩ��֮��
gain = (k_i_in/(2*m))-(sum_tot*k_i/(2*m^2));  % ���ݹ�ʽ����ģ�������
end