clear
clc

N_test=5000;
x_test=zeros(N_test,6);
for i=1:N_test
    for j=1:6
        x_test(i,j)=floor(rand*10);
    end
end

save x_test.mat x_test