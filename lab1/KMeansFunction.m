function [pattern,centers,silhouette] = KMeansFunction(K,data)
[n,m] = size(data);
pattern = zeros(n,m+1);
pattern(:,1:m) = data(:,:);

centers = zeros(K,m);
centers(1,:) = data(randi(n),:); 
for i = 2:K
    temp_distance = zeros(n,i-1);
    temp_min = zeros(1,n);
    for j = 1:n
        for k = 1:i-1
            temp_distance(j,k) = norm(data(j,:)-centers(k,:));
        end
    end
    for j = 1:n
        temp_min(j) =  min(temp_distance(j,:));
    end
    [~,index] = max(temp_min);
    centers(i,:) = data(index,:);
end

for t = 1: 500000
    distance = zeros(n,K);
    num = zeros(1,K);
    new_centers = zeros(K,m);
    
    for i = 1:n
        for j = 1:K
            distance(i,j) = norm(data(i,:)-centers(j,:));
        end
    end
    for i = 1:n
        [~,index] =  min(distance(i,:));
        pattern(i,m+1) = index;
    end
    
    for i = 1:K
        for j = 1:n
            if pattern(j,m+1) == i
                new_centers(i,:) = new_centers(i,:)+pattern(j,1:m);
                num(i) = num(i)+1;
            end
        end
        new_centers(i,:) = new_centers(i,:)/num(i);
    end
    
    if new_centers == centers
        break;
    else
        centers = new_centers;
    end
end

% ����������
a = zeros(n,1);     % ���ڲ����ƶ�
b = zeros(n,K-1);   % �ؼ䲻���ƶ�
s = zeros(n,1);     % �����������ϵ��
silhouette = 0;     % ������������ϵ��
for i = 1:n
    same_num = 0;
    diff_num = zeros(1,K-1);
    % ���������㣬����ÿ��������Ĵ��ڲ����ƶȺʹؼ䲻���ƶ�
    for j = 1:n     
       if pattern(i,m+1) == pattern(j,m+1) 
           a(i) = a(i)+norm(data(i,:)-data(j,:));
           same_num = same_num + 1;
       else
           for k = 1:K
               if pattern(j,m+1) == k && k < pattern(i,m+1)
                   b(i,k) = b(i,k)+norm(data(i,:)-data(j,:));
                   diff_num(k) = diff_num(k)+1;
               elseif pattern(j,m+1) == k && k > pattern(i,m+1)
                   b(i,k-1) = b(i,k-1)+norm(data(i,:)-data(j,:));
                   diff_num(k-1) = diff_num(k-1)+1;
               end
           end
       end
    end
    % ���ڲ����ƶ�Ϊ���������������������ƽ������
    a(i) = a(i)/same_num;
    % �ؼ䲻���ƶ�Ϊ�������벻ͬ����������ƽ���������Сֵ
    for j = 1:K-1
        b(i,j) = b(i,j)/diff_num(j);
    end
    bi = min(b(i,:));
    % ��������������ϵ��
    if a(i) > bi
        s(i) = (bi-a(i))/a(i);
    else
        s(i) = (bi-a(i))/bi;
    end
end
% ������������ϵ��Ϊ����������ϵ����ƽ��ֵ
for i = 1:n
    silhouette = silhouette+s(i);
end
silhouette = silhouette/n;
end

