clear
clc

%% Parameters
n=10;
N_iter=20;

%% Set test point
N_test=5000;
load('x_test.mat')
x_test=x_test+1;
y_real=zeros(N_test,N_iter+1);

%% Initial policy
temp_fx=[0,0,0,1,-1];
temp_fy=[0,1,-1,0,0];

J=zeros(n,n,n,n,n,n);
for s1x=1:n
    for s1y=1:n
        for s2x=1:n
            for s2y=1:n
                for fx=1:n
                    for fy=1:n
                        dis1=abs(s1x-fx)+abs(s1y-fy);
                        dis2=abs(s2x-fx)+abs(s2y-fy);
                        J(s1x,s1y,s2x,s2y,fx,fy)=min(dis1,dis2);
                    end
                end
            end
        end
    end
end

for i=1:N_test
        y_real(i,1)=J(x_test(i,1),x_test(i,2),x_test(i,3),x_test(i,4),x_test(i,5),x_test(i,6));
end

%% policy iteration
temp_sx=[0,0,0,1,-1];
temp_sy=[0,1,-1,0,0];
temp_fx=[0,0,0,1,-1];
temp_fy=[0,1,-1,0,0];


for o=1:N_iter
    J1=zeros(n,n,n,n,n,n);
    for s1x=1:n
        for s1y=1:n
            for s2x=1:n
                for s2y=1:n
                    for fx=1:n
                        for fy=1:n
                            dis1=abs(s1x-fx)+abs(s1y-fy);
                            dis2=abs(s2x-fx)+abs(s2y-fy);
                            if dis1==0 || dis2==0   % if the distance equal to zero then no renew
                                J1(s1x,s1y,s2x,s2y,fx,fy)=0;
                            else
                                cost_temp=[];
                                for c1=1:5  % control1
                                    for c2=1:5  %control2
                                        xt1=s1x+temp_sx(c1);
                                        xt2=s2x+temp_sx(c2);
                                        yt1=s1y+temp_sy(c1);
                                        yt2=s2y+temp_sy(c2);
                                        if xt1>0 && xt1<=n && xt2>0 && xt2<=n && yt1>0 && yt1<=n && yt2>0 && yt2<=n
                                            rand_sum=[];
                                            for r=1:5
                                                xr1=fx+temp_fx(r);
                                                yr1=fy+temp_fy(r);
                                                if xr1>0 && xr1<=n && yr1>0 && yr1<=n
                                                    rand_sum=[rand_sum J(xt1,yt1,xt2,yt2,xr1,yr1)];
                                                end
                                            end
                                            cost_temp=[cost_temp mean(rand_sum)];
                                        end
                                    end % end control 1
                                end % end control 2
                                J1(s1x,s1y,s2x,s2y,fx,fy)=1+min(cost_temp);
                                
                            end
                        end
                    end
                end
            end
         disp([o,s1x,s1y])
        end
    end
    J=J1;
    
    for i=1:N_test
        y_real(i,o+1)=J(x_test(i,1),x_test(i,2),x_test(i,3),x_test(i,4),x_test(i,5),x_test(i,6));
    end
    %disp(o)
end
    
save('policy.mat','n','J')
save('y_real.mat','y_real')