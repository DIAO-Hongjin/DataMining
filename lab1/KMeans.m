% ��������
data1 = load('Aggregation_cluster=7.txt');
data2 = load('flame_cluster=2.txt');
data3 = load('Jain_cluster=2.txt');
data4 = load('Pathbased_cluster=3.txt');
data5 = load('Spiral_cluster=3.txt');
data6 = load('Mfeat.mat');

% ԭʼ����
% data = data5;
% data = zscore(data6.data_mor);  % ��ȡdata_mor���������й�һ��
% data = zscore(data6.data_zer);  % ��ȡdata_zer���������й�һ��
% data = zscore(data6.data_pix);  % ��ȡdata_pix���������й�һ��
% data = zscore(data6.data_kar);  % ��ȡdata_kar���������й�һ��
% data = zscore(data6.data_fou);  % ��ȡdata_mor���������й�һ��
data = zscore(data6.data_fac);  % ��ȡdata_fou���������й�һ��

K = 10;
[n,m] = size(data);
pattern = zeros(n,m+1);
pattern(:,1:m) = data(:,:);

% ���ĳ�ʼ��
% ԭʼ���������ѡ��K����������Ϊ����
% centers = zeros(K,m);
% for i = 1:K   
%     centers(i,:) = data(randi(n),:);  
% end

% ���ĳ�ʼ��
% K-means++�������д��ĵ���С����Խ���������Խ�п��ܱ�ѡ��Ϊ����
% centers = zeros(K,m);
% centers(1,:) = data(randi(n),:);    % ���ѡ���һ������
% for i = 2:K    %ѡ��ʣ��Ĵ���
%     temp_distance = zeros(n,i-1);   % ��¼�������뵱ǰ���ĵľ���
%     temp_min = zeros(1,n);          % ��¼�������뵱ǰ���ĵ���С���� 
%     distance_sum = 0;               % ��¼�������뵱ǰ���ĵ���С����֮��
%     for j = 1:n     % �����������뵱ǰ���ĵľ���
%         for k = 1:i-1
%             temp_distance(j,k) = norm(data(j,:)-centers(k,:));
%         end
%     end
%     for j = 1:n     % �����������뵱ǰ���ĵ���С����
%         temp_min(j) =  min(temp_distance(j,:));
%         distance_sum = distance_sum+temp_min(j);
%     end
%     % ʹ�����̶��㷨���Ծ���Ϊ����ѡ����Ϊ��һ�����ĵ�������
%     random_num = distance_sum*rand; % �൱�����̵�ָ��
%     for j = 1:n     % ����������Ĵ�Сѡ���Ӧ��������Ϊ��һ������
%         random_num = random_num - temp_min(j);
%         if random_num <= 0     % ָ����ָ��������
%             centers(i,:) = data(j,:);
%             break;
%         end
%     end
% end

% ���ĳ�ʼ��
% �����д��ĵ���С�������������㱻ѡ��Ϊ����
centers = zeros(K,m);
centers(1,:) = data(randi(n),:);    % ���ѡ���һ������
for i = 2:K    %ѡ��ʣ��Ĵ���
    temp_distance = zeros(n,i-1);   % ��¼�������뵱ǰ���ĵľ���
    temp_min = zeros(1,n);          % ��¼�������뵱ǰ���ĵ���С����
    for j = 1:n     % �����������뵱ǰ���ĵľ���
        for k = 1:i-1
            temp_distance(j,k) = norm(data(j,:)-centers(k,:));
        end
    end
    for j = 1:n     % �����������뵱ǰ���ĵ���С����
        temp_min(j) =  min(temp_distance(j,:));
    end
    % �뵱ǰ���ĵ���С�������������㱻ѡ��Ϊ��һ������
    [temp_max, index] = max(temp_min);
    centers(i,:) = data(index,:);
end

 % plot(data(:,1),data(:,2),'+',centers(:,1),centers(:,2),'ko');

% Ŀ�꺯�����Ż�����
for t = 1: 10000    % ����������Ϊ10000��
    distance = zeros(n,K);      % ��¼ÿ����������ÿ�����ĵľ���
    num = zeros(1,K);           % ��¼ÿ�����������������
    new_centers = zeros(K,m);   % ��¼�µĴ���

    for i = 1:n     % ������������ÿ�����ĵľ���
        for j = 1:K
            distance(i,j) = norm(data(i,:)-centers(j,:));
        end
    end
    for i = 1:n     % �������㻮�ֵ�����Ĵ��������������
        [min_distance,index] =  min(distance(i,:));
        pattern(i,m+1) = index;
    end
    
    for i = 1:K     % ���¼������
        for j = 1:n
            if pattern(j,m+1) == i
                new_centers(i,:) = new_centers(i,:)+pattern(j,1:m);
                num(i) = num(i)+1;
            end
        end
        new_centers(i,:) = new_centers(i,:)/num(i);
    end
    
    if new_centers == centers  % ���Ĳ��ٱ仯��˵���������˳�ѭ��
        break;
    else                       % ��δ���������´��ģ��������� 
        centers = new_centers;
    end
end

% �˹����ݼ�����ͼ����������
% figure;  
% hold on;  
% for i = 1:n 
%     for j = 1:K
%         if pattern(i,m+1) == j
%             if j == 1
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[1 0 0]);
%             elseif j == 2
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0 1 0]);
%             elseif j == 3
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0 0 1]);
%             elseif j == 4
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[1 0 1]);
%             elseif j == 5
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[1 1 0]);
%             elseif j == 6
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0 0 0]);
%             elseif j == 7
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0 1 1]);
%             elseif j == 8
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0 0.5 0.5]);
%             elseif j == 9
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0.5 0.5 0]);
%             elseif j == 10
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0.5 0 0.5]);
%             end
%             plot(centers(j,1),centers(j,2),'ko');
%         end
%     end
% end  
% grid on;  

% ��ʵ���ݼ� ���������
result = ClusteringMeasure(pattern(:,m+1),data6.classid)
