clc
clear all
%% read data and process data
[test,test_yy,train,train_yy]=data_process;

%% train the model using T-S FUZZY METHOD
% calculate the center and membership
clus_num=4;              % suppose we clus the data into 3 clusters.
iter_time=50;            % how many time run for iteration
pow=2;                   % power of u
[u,c]=FC_function(train,clus_num,iter_time,pow);

%figure the cluster results
figure;
[~,label] = max(u,[],2); 
gscatter(train(:,1),train(:,2),label)
saveas(gcf,"fuzzy cluster.jpg")

% extend the regressor matrix
leg_tra=size(train,1);
wig_tra=size(train,2);
train_ext=ones(leg_tra,wig_tra+1);
train_ext(:,2:wig_tra+1)=train;

% weighted batch least square to calculate the regressor coeffients
for i=1:clus_num
    D=diag(u(:,i));
    D_q=D.^pow;
    the(:,i)=(train_ext'*D_q*train_ext)^-1*train_ext'*D_q*train_yy;
end

%calculate the yhat
for j=1:leg_tra
    for i=1:clus_num
        y(j,i)=u(j,i)*train_ext(j,:)*the(:,i);
    end
    yhat(j)=sum(y(j,:));
end
plot(1:leg_tra,yhat,'-',1:leg_tra,train_yy)
saveas(gcf,"model_building.jpg")

%% validate model
% calculate the membership of every test sample
[u_test,c]=mmb_function(test,clus_num,iter_time,pow,c);
% extend the test regressor matrix
leg_test=size(test,1);
wig_test=size(test,2);
test_ext=ones(leg_test,wig_test+1);
test_ext(:,2:wig_test+1)=test;

%calculate the yhat for test value
for j=1:leg_test
    for i=1:clus_num
        y_t(j,i)=u_test(j,i)*test_ext(j,:)*the(:,i);
    end
    yhat_t(j)=sum(y_t(j,:));
end
plot(1:leg_test,yhat_t,'-',1:leg_test,test_yy)
saveas(gcf,"model_validation.jpg")






%% data read and process function============================ 
function [test,test_yy,train,train_yy,leg_tra]=data_process
train_yy=dlmread("trainyy.txt"); % read the train set
train_uu=dlmread("trainuu.txt"); % read the train set
test_yy=dlmread("testyy.txt"); % read the train set
test_uu=dlmread("testuu.txt"); % read the train set

train=[train_yy,train_uu];
leg_tra=size(train,1);
train(leg_tra,:)=[];
train_yy(1)=[];

test=[test_yy,test_uu];
leg_test=size(test,1);
test(leg_test,:)=[];
test_yy(1)=[];
end


%% fuzzy cluster function======================================
function [u,c]=FC_function(data,clus_num,iter_time,pow)
dat_dim=size(data,2); % figure out the dimesion of data
dat_num=size(data,1); % figure out the total number of data

%% initial the U data
u = rand(clus_num,dat_num);
u_col_sum=sum(u);
for m=1:dat_num
    u(:,m)=u(:,m)/u_col_sum(m);
end
u=u';  %row: dim_data; column: cluster number

%% iteration for clustering
for r=1:1:iter_time % how many round(r) we run for iteration
    %% iteration for center(cluster)
   
    um=u.^pow; % calculate the u_power(m)
    sum_um_col=sum(um);
    sum_datum=data'*um;
    for clu=1:clus_num
         c(:,clu)=sum_datum(:,clu)/sum_um_col(clu);
    end
   
    %% iteration for U (membership function)
    c=c';
    for j = 1:clus_num
        for k = 1:dat_num
            sum1 = 0;
            for j1 = 1:clus_num
                temp = (norm(data(k,:)-c(j,:))/norm(data(k,:)-c(j1,:))).^(2/(pow-1));
                sum1 = sum1 + temp;
            end
            u(k,j) = 1./sum1;
        end
    end
    c=c';
end %  end for iteration for clustering
end

%% membership function ======================================
function [u,c]=mmb_function(data,clus_num,iter_time,pow,c)
dat_dim=size(data,2); % figure out the dimesion of data
dat_num=size(data,1); % figure out the total number of data
    c=c';
    for j = 1:clus_num
        for k = 1:dat_num
            sum1 = 0;
            for j1 = 1:clus_num
                temp = (norm(data(k,:)-c(j,:))/norm(data(k,:)-c(j1,:))).^(2/(pow-1));
                sum1 = sum1 + temp;
            end
            u(k,j) = 1./sum1;
        end
    end
    c=c';
end