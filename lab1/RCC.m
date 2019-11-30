% 数据输入
data1 = load('Aggregation_cluster=7.txt');
data2 = load('flame_cluster=2.txt');
data3 = load('Jain_cluster=2.txt');
data4 = load('Pathbased_cluster=3.txt');
data5 = load('Spiral_cluster=3.txt');
data6 = load('Mfeat.mat');

% 原始数据
data = data1;
% data = zscore(data6.data_mor);  % 读取data_mor特征并进行归一化

[n,m] = size(data);

% 构造K近邻图作为相似度矩阵
K = 12;  
epsilon = zeros(n,n); 
DistanceMaztrix = pdist2(data,data);  % 计算距离矩阵
for i = 1:n
    [distance_order, index_order] = sort(DistanceMaztrix(i,:));
    % i为j的K近邻或j为i的K近邻时，i与j之间存在一条边，权值为距离
    for j = 2:K+1  
        epsilon(i,index_order(j)) = distance_order(j);
        epsilon(index_order(j),i) = distance_order(j);
    end
end

xi = norm(data);

% 初始化W_pq
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

% 初始化delta
% delta为图epsilon中最短的1%边的平均长度
rate = 0.01;
edges = [];
for i = 1:n  % 找到无向图中的所有边
    for j = (i+1):n
        if epsilon(i,j) ~= 0
            edges = [edges epsilon(i,j)];
        end
    end
end
edges_num = numel(edges);
[edges_order,~] = sort(edges);  % 对边按长度进行排序
% 取最短的1%的边，取平均长度
delta = sum(edges_order(1:round(rate*edges_num)))/round(rate*edges_num);

% 初始化代表点U、l_pq和mu
U = data;                    % U初始化为样本点
l_pq = ones(n);              % l_pq初始化为1
mu = 3*max(max(epsilon))^2;  % mu = 3r^2，r为图epsilon中最长的边

% 初始化lambda
I = eye(n);
A = zeros(n);
for i = 1:n              % 计算A
    for j = 1:n
        A = A+W_pq(i,j)*l_pq(i,j)*(I(:,i)-I(:,j))*(I(:,i)-I(:,j))';
    end
end
lambda = xi/norm(A);     % 计算lambda的初始值

% 计算初始时目标函数的值
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

% 优化目标函数
for i = 2:(max_iterators+1)
    for p = 1:n             % 更新l_pq
        for q = 1:n
            l_pq(p,q) = (mu/(mu+norm(U(p,:)-U(q,:))^2))^2;
        end
    end
    
    A = zeros(n);
    for p = 1:n             % 更新A
        for q = 1:n
            A = A+W_pq(p,q)*l_pq(p,q)*(I(:,p)-I(:,q))*(I(:,p)-I(:,q))';
        end
    end
    
    M = I+lambda*A;
    U = (data'*(M^-1))';    % 更新U
    if mod(i-1,4) == 0      % 更新lambda和mu
        lambda = xi/norm(A);
        if mu >= delta
            mu = mu/2;            
        else
            mu = delta/2;  
        end
    end
    
    % 计算更新后的目标函数
    first_term = 0.5*sum(sum((data-U).^2));
    second_term = 0;
    for p = 1:n
        for q = 1:n
            second_term = second_term+W_pq(p,q)*(l_pq(p,q)*norm(U(p,:)-U(q,:))^2+mu*(sqrt(l_pq(p,q))-1)^2);
        end
    end
    second_term = lambda*0.5*second_term;
    objective(i) = first_term+second_term;
    
    if abs(objective(i)-objective(i-1)) >= 0.1  % 判断是否达到停止条件
        break;
    end
end

% figure;
% plot(data(:,1),data(:,2),'.');
figure;
plot(U(:,1),U(:,2),'.');

% 构造结果图
final_graph = zeros(n);
% 如果代表点的距离小于delta，则添加一条边
for i = 1:n
    for j = (i+1):n
        if norm(U(i,:)-U(j,:)) < delta
            final_graph(i,j) = 1;
            final_graph(j,i) = 1;
        end
    end
end
% 结果图的连通分量即为簇的划分结果
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