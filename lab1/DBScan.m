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

[n,m] = size(data);

Eps = 4;
MinPts = 48;

% �ܶ���ֵ
% MinPts = 7;    
% % ����뾶
% Eps = ((prod(max(data)-min(data))*MinPts*gamma(.5*m+1))/(n*sqrt(pi.^m))).^(1/m);

% ʶ�������������
% patternΪDBSCANģ�ͣ�n�������㣬m��ά��
% ��m+1ά��������������ࣺ-1��0��1�ֱ���������㡢�߽��ͺ��ĵ�
% ��m+2ά���������������Ĵأ�-1��0��K�ֱ���������㡢δ��������K��
pattern = zeros(n,m+2);         
pattern(:,1:m) = data(:,:);
Pts = zeros(1,n);                       % ��������ܶ�
DistanceMaztrix = pdist2(data,data);    % ���ݼ��ľ������
for i = 1:n     % ���������ȱ��Ϊ������
    pattern(i,m+1) = -1;
end
for i = 1:n     % ����ÿ������������򣬼�¼��������ܶ�
    % ��¼����Χ�ڵ����������
    neighbors = find(DistanceMaztrix(i,:)<=Eps);  
    Pts(i) = numel(neighbors);
    % ���ܶȴ����ܶ���ֵ�����������������Ϊ���ĵ�
    % �Һ��ĵ������Χ�ڵķǺ�����������������Ϊ�߽��
    if Pts(i) >= MinPts
        pattern(i,m+1) = 1;             % ���ĵ�
        for j = 1:Pts(i)
            if pattern(neighbors(j),m+1) ~= 1
                pattern(neighbors(j),m+1) = 0; % �߽��
            end
        end
    end
end

% figure;  
% hold on;  
% for i = 1:n
%     if pattern(i,m+1) == 1
%         plot(pattern(i,1),pattern(i,2),'.','Color',[1 0 0]);
%     elseif pattern(i,m+1) == 0
%         plot(pattern(i,1),pattern(i,2),'.','Color',[0 1 0]);
%     elseif pattern(i,m+1) == -1
%         plot(pattern(i,1),pattern(i,2),'.','Color',[0 0 1]);
%     else
%         plot(pattern(i,1),pattern(i,2),'.','Color',[0 0 0]);
%     end
% end

% �������
K = 0;     % ��¼�ص�����
for i = 1:n     % ����������
    if pattern(i,m+2) == 0               % ����δ��������������
        if pattern(i,m+1) == 1           % δ�������ĺ��ĵ�
            K = K+1;                     % �ص�������һ
            pattern(i,m+2) = K;          % ���δ�������ĺ��ĵ����Ϊ�µĴ�
            neighbors = find(DistanceMaztrix(i,:)<=Eps);     % ��������
            cnt = 1;
            while true
                j = neighbors(cnt);
                % �����ڵĵ�����ڴ�K
                % �������ڴ���δ�������ĺ��ĵ㣬���������������ĵ������
                if pattern(j,m+2) == 0
                    pattern(j,m+2) = K;
                    if pattern(j,m+1) == 1     
                        neighbors = [neighbors find(DistanceMaztrix(j,:)<=Eps)];
                    end
                end
                cnt = cnt+1;
                if cnt > numel(neighbors) % �����������
                    break;
                end
            end
        elseif pattern(i,m+1) == -1      % ������Ϊ�����㣬���������ҲΪ������
            pattern(i,m+2) = -1;
        end
    end
end

% figure;  
% hold on;  
% for i = 1:n 
%     for j = -1:K
%         if pattern(i,m+2) == j
%             if j == -1
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0 0 0]);
%             elseif j == 1
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
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0 1 1]);
%             elseif j == 7
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0.5 0.5 0.5]);
%             elseif j == 8
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0 0.5 0.5]);
%             elseif j == 9
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0.5 0.5 0]);
%             elseif j == 10
%                 plot(pattern(i,1),pattern(i,2),'.','Color',[0.5 0 0.5]);
%             end
%         end
%     end
% end  
% grid on;  

result = ClusteringMeasure(pattern(:,m+2),data6.classid)