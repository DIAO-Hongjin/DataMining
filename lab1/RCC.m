% ��������
data1 = load('Aggregation_cluster=7.txt');
data2 = load('flame_cluster=2.txt');
data3 = load('Jain_cluster=2.txt');
data4 = load('Pathbased_cluster=3.txt');
data5 = load('Spiral_cluster=3.txt');
data6 = load('Mfeat.mat');

% ԭʼ����
data = data1;
% data = zscore(data6.data_mor);  % ��ȡdata_mor���������й�һ��

[n,m] = size(data);

% ����K����ͼ��Ϊ���ƶȾ���
K = 12;  
epsilon = zeros(n,n); 
DistanceMaztrix = pdist2(data,data);  % ����������
for i = 1:n
    [distance_order, index_order] = sort(DistanceMaztrix(i,:));
    % iΪj��K���ڻ�jΪi��K����ʱ��i��j֮�����һ���ߣ�ȨֵΪ����
    for j = 2:K+1  
        epsilon(i,index_order(j)) = distance_order(j);
        epsilon(index_order(j),i) = distance_order(j);
    end
end

xi = norm(data);

% ��ʼ��W_pq
W_pq = zeros(n,n); 
total_degree = numel(find(epsilon));
for i = 1:n
    for j = (i+1):n
        if epsilon(i,j) ~= 0
            W_pq(i,j) = total_degree/(n*sqrt(numel(find(epsilon(i,:)))*numel(find(epsilon(j,:)))));
            W_pq(j,i) = total_degree/(n*sqrt(numel(find(epsilon(i,:)))*numel(find(epsilon(j,:)))));
        end
    end
end

% ��ʼ��delta
% deltaΪͼepsilon����̵�1%�ߵ�ƽ������
rate = 0.01;
edges = [];
for i = 1:n  % �ҵ�����ͼ�е����б�
    for j = (i+1):n
        if epsilon(i,j) ~= 0
            edges = [edges epsilon(i,j)];
        end
    end
end
edges_num = numel(edges);
[edges_order,~] = sort(edges);  % �Ա߰����Ƚ�������
% ȡ��̵�1%�ıߣ�ȡƽ������
delta = sum(edges_order(1:round(rate*edges_num)))/round(rate*edges_num);

% ��ʼ�������U��l_pq��mu
U = data;                    % U��ʼ��Ϊ������
l_pq = ones(n);              % l_pq��ʼ��Ϊ1
mu = 3*max(max(epsilon))^2;  % mu = 3r^2��rΪͼepsilon����ı�

% ��ʼ��lambda
I = eye(n);
A = zeros(n);
for i = 1:n              % ����A
    for j = 1:n
        A = A+W_pq(i,j)*l_pq(i,j)*(I(:,i)-I(:,j))*(I(:,i)-I(:,j))';
    end
end
lambda = xi/norm(A);     % ����lambda�ĳ�ʼֵ

% �����ʼʱĿ�꺯����ֵ
max_iterators = 100;
objective = zeros(1,max_iterators+1);
first_term = 0.5*sum(sum((data-U).^2));
second_term = 0;
for i = 1:n
    for j = 1:n
        second_term = second_term+W_pq(i,j)*(l_pq(i,j)*norm(U(i,:)-U(j,:))^2+mu*(sqrt(l_pq(i,j))-1)^2);
    end
end
second_term = lambda*0.5*second_term;
objective(1) = first_term+second_term;

% �Ż�Ŀ�꺯��
for i = 2:(max_iterators+1)
    for p = 1:n             % ����l_pq
        for q = 1:n
            l_pq(p,q) = (mu/(mu+norm(U(p,:)-U(q,:))^2))^2;
        end
    end
    
    A = zeros(n);
    for p = 1:n             % ����A
        for q = 1:n
            A = A+W_pq(p,q)*l_pq(p,q)*(I(:,p)-I(:,q))*(I(:,p)-I(:,q))';
        end
    end
    
    M = I+lambda*A;
    U = (data'*(M^-1))';    % ����U
    if mod(i-1,4) == 0      % ����lambda��mu
        lambda = xi/norm(A);
        if mu >= delta
            mu = mu/2;            
        else
            mu = delta/2;  
        end
    end
    
    % ������º��Ŀ�꺯��
    first_term = 0.5*sum(sum((data-U).^2));
    second_term = 0;
    for p = 1:n
        for q = 1:n
            second_term = second_term+W_pq(p,q)*(l_pq(p,q)*norm(U(p,:)-U(q,:))^2+mu*(sqrt(l_pq(p,q))-1)^2);
        end
    end
    second_term = lambda*0.5*second_term;
    objective(i) = first_term+second_term;
    
    if abs(objective(i)-objective(i-1)) >= 0.1  % �ж��Ƿ�ﵽֹͣ����
        break;
    end
end

% figure;
% plot(data(:,1),data(:,2),'.');
figure;
plot(U(:,1),U(:,2),'.');

% ������ͼ
final_graph = zeros(n);
% ��������ľ���С��delta�������һ����
for i = 1:n
    for j = (i+1):n
        if norm(U(i,:)-U(j,:)) < delta
            final_graph(i,j) = 1;
            final_graph(j,i) = 1;
        end
    end
end
% ���ͼ����ͨ������Ϊ�صĻ��ֽ��
label = conncomp(graph(final_graph));

% figure;
% plot(graph(final_graph));
figure;  
hold on;  
for i = 1:n 
    for j = 1:numel(unique(label))
        if label(i) == j
            if j == 1
                plot(data(i,1),data(i,2),'.','Color',[1 0 0]);
            elseif j == 2
                plot(data(i,1),data(i,2),'.','Color',[0 1 0]);
            elseif j == 3
                plot(data(i,1),data(i,2),'.','Color',[0 0 1]);
            elseif j == 4
                plot(data(i,1),data(i,2),'.','Color',[1 0 1]);
            elseif j == 5
                plot(data(i,1),data(i,2),'.','Color',[1 1 0]);
            elseif j == 6
                plot(data(i,1),data(i,2),'.','Color',[0 0 0]);
            elseif j == 7
                plot(data(i,1),data(i,2),'.','Color',[0 1 1]);
            elseif j == 8
                plot(data(i,1),data(i,2),'.','Color',[0 0.5 0.5]);
            elseif j == 9
                plot(data(i,1),data(i,2),'.','Color',[0.5 0.5 0]);
            elseif j == 10
                plot(data(i,1),data(i,2),'.','Color',[0.5 0 0.5]);
            elseif j == 11
                plot(data(i,1),data(i,2),'.','Color',[0.2 0 0]);
            elseif j == 12
                plot(data(i,1),data(i,2),'.','Color',[0 0.2 0]);
            elseif j == 13
                plot(data(i,1),data(i,2),'.','Color',[0 0 0.2]);
            elseif j == 14
                plot(data(i,1),data(i,2),'.','Color',[0.2 0 0.2]);
            elseif j == 15
                plot(data(i,1),data(i,2),'.','Color',[0.2 0.2 0]);
            elseif j == 16
                plot(data(i,1),data(i,2),'.','Color',[0 0.2 0.2]);
            elseif j == 17
                plot(data(i,1),data(i,2),'.','Color',[0.7 0 0]);
            elseif j == 18
                plot(data(i,1),data(i,2),'.','Color',[0 0.7 0]);
            elseif j == 19
                plot(data(i,1),data(i,2),'.','Color',[0 0 0.7]);
            elseif j == 20
                plot(data(i,1),data(i,2),'.','Color',[0.7 0 0.7]);
            elseif j == 21
                plot(data(i,1),data(i,2),'.','Color',[0.7 0.7 0]);
            elseif j == 22
                plot(data(i,1),data(i,2),'.','Color',[0 0.7 0.7]);
            elseif j == 23
                plot(data(i,1),data(i,2),'.','Color',[0.7 0.7 0.7]);
            elseif j == 24
                plot(data(i,1),data(i,2),'.','Color',[0.2 0.5 0.7]);
            elseif j == 25
                plot(data(i,1),data(i,2),'.','Color',[0.7 0.5 0.2]);
            elseif j == 26
                plot(data(i,1),data(i,2),'.','Color',[0.5 0.7 0.2]);
            elseif j == 27
                plot(data(i,1),data(i,2),'.','Color',[0.5 0.2 0.7]);
            elseif j == 28
                plot(data(i,1),data(i,2),'.','Color',[0.2 0.7 0.5]);
            elseif j == 29
                plot(data(i,1),data(i,2),'.','Color',[0.7 0.2 0.5]);
            elseif j == 30
                plot(data(i,1),data(i,2),'.','Color',[0.7 0.2 0.2]);
            elseif j == 31
                plot(data(i,1),data(i,2),'.','Color',[0.7 0.5 0.5]);
            elseif j == 32
                plot(data(i,1),data(i,2),'.','Color',[0.2 0.2 0.7]);
            elseif j == 33
                plot(data(i,1),data(i,2),'.','Color',[0.2 0.7 0.2]);
            elseif j == 34
                plot(data(i,1),data(i,2),'.','Color',[0.5 0.7 0.5]);
            elseif j == 35
                plot(data(i,1),data(i,2),'.','Color',[0.5 0.5 0.7]);
            elseif j == 36
                plot(data(i,1),data(i,2),'.','Color',[0.2 0.7 0.7]);
            elseif j == 37
                plot(data(i,1),data(i,2),'.','Color',[0.7 0.2 0.7]);
            elseif j == 38
                plot(data(i,1),data(i,2),'.','Color',[0.7 0.7 0.2]);
            elseif j == 39
                plot(data(i,1),data(i,2),'.','Color',[0.7 0.7 0.5]);
            elseif j == 40
                plot(data(i,1),data(i,2),'.','Color',[0.5 0.7 0.7]);
            end
        end
    end
end  
grid on;  

% result = ClusteringMeasure(label,data6.classid)