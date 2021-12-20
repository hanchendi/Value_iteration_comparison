clear
clc

load('y_real.mat')
load('y_test.mat')

for i=20:25
    plot(0:20,y_test(i,:));hold on
    plot(0:20,y_real(i,:),'--');hold on
end