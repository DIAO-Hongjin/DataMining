% ��������
data1 = load('Aggregation_cluster=7.txt');
data2 = load('flame_cluster=2.txt');
data3 = load('Jain_cluster=2.txt');
data4 = load('Pathbased_cluster=3.txt');
data5 = load('Spiral_cluster=3.txt');
data6 = load('Mfeat.mat');

% ԭʼ����
% data = data2;
% data = zscore(data6.data_mor);  % ��ȡdata_mor���������й�һ��
% data = zscore(data6.data_zer);  % ��ȡdata_zer���������й�һ��
% data = zscore(data6.data_pix);  % ��ȡdata_pix���������й�һ��
% data = zscore(data6.data_kar);  % ��ȡdata_kar���������й�һ��
% data = zscore(data6.data_fou);  % ��ȡdata_mor���������й�һ��
data = zscore(data6.data_fac);  % ��ȡdata_fou���������й�һ��

[n,m] = size(data);

% ѡȡ�ضϾ���
percent = 3;  % ����ÿ����������������������ƽ��ֵռ���������ı���
DistanceMaztrix = pdist2(data,data);  % �������
all_distance = [];
for i = 1:n   % ��¼���о���ֵ
    all_distance = [all_distance DistanceMaztrix(i,(i+1):n)]; 
end
[distance_order, ~] = sort(all_distance);  % �Ծ���ֵ��������
radius = distance_order(round(percent/100*n^2/2));  %���ձ���ѡ��ضϾ���

%radius = 1.3086;  
% ����ֲ��ܶ�
% Cut-off kernel�������ڵ�����������
% rou = zeros(1,n);
% for i = 1:n
%     for j = 1:n
%         if DistanceMaztrix(i,j) <= radius
%             rou(i) = rou(i) + 1;
%         end
%     end
% end
% ����ֲ��ܶ�
% Gaussian kernel
rou = zeros(1,n);
for i = 1:n
    for j = 1:n
        if i ~= j
            rou(i) = rou(i) + exp(-(DistanceMaztrix(i,j)/radius)^2);
        end
    end
end

% ������Ծ���
delta = zeros(1,n);
neighbor = zeros(1,n);
for i = 1:n
    [distance_order, index_order] = sort(DistanceMaztrix(i,:)); % ����
    for j = 1:n
        if rou(index_order(j)) > rou(i)  % �ҵ���������ľֲ��ܶȸ��ߵ�������
            delta(i) = distance_order(j);
            neighbor(i) = index_order(j);
            break;
        end
    end
    if delta(i) == 0  %��������Ϊȫ���ܶ���ߣ�����Ծ���Ϊ������
        delta(i) = distance_order(n);
    end
end

% ��������ͼ
figure(10000); 
plot(rou,delta,'.');
title ('Decision Graph')
xlabel ('\rho')
ylabel ('\delta')
rect = getrect(10000);  % �����Ӿ���ͼ��ѡȡ����
rou_min = rect(1);
delta_min = rect(2);
close all;

% ȷ������
K = 0;                % ��¼�ص�����
center_index = [];    % ��¼����
centers = zeros(K,m);
for i = 1:n  % �ֲ��ܶȺ���Ծ�����������ѡȡ����ֵʱ���õ㱻ѡ��Ϊ����
    if rou(i) > rou_min && delta(i) > delta_min
        K = K+1;
        center_index = [center_index i];
    end
end

% �������
label = zeros(1,n);                  % ��¼�������������
[~,rou_index] = sort(rou,'descend'); % �ֲ��ܶȴӸߵ�������
for i = 1:K      % ��ȷ���������������
    label(center_index(i)) = i;
end
for i = 1:n      % ���������Ϊ����������ܶȸ��ߵĵ�����
    if label(rou_index(i)) == 0
        label(rou_index(i)) = label(neighbor(rou_index(i)));
    end
end

% figure;  
% hold on;  
% for i = 1:n 
%     for j = 1:K
%         if label(i) == j
%             if j == 1
%                 plot(data(i,1),data(i,2),'.','Color',[1 0 0]);
%             elseif j == 2
%                 plot(data(i,1),data(i,2),'.','Color',[0 1 0]);
%             elseif j == 3
%                 plot(data(i,1),data(i,2),'.','Color',[0 0 1]);
%             elseif j == 4
%                 plot(data(i,1),data(i,2),'.','Color',[1 0 1]);
%             elseif j == 5
%                 plot(data(i,1),data(i,2),'.','Color',[1 1 0]);
%             elseif j == 6
%                 plot(data(i,1),data(i,2),'.','Color',[0 0 0]);
%             elseif j == 7
%                 plot(data(i,1),data(i,2),'.','Color',[0 1 1]);
%             elseif j == 8
%                 plot(data(i,1),data(i,2),'.','Color',[0 0.5 0.5]);
%             elseif j == 9
%                 plot(data(i,1),data(i,2),'.','Color',[0.5 0.5 0]);
%             elseif j == 10
%                 plot(data(i,1),data(i,2),'.','Color',[0.5 0 0.5]);
%             end
%         end
%         plot(data(center_index(j),1),data(center_index(j),2),'ko');
%     end
% end  
% grid on;  

result = ClusteringMeasure(label,data6.classid)