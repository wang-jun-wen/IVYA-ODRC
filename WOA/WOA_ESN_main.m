%% 清空环境变量
close all    %关闭开启的窗口
clear        %清空变量
clc          %清空命令行 
warning('off', 'MATLAB:nearlySingularMatrix');
%% 训练-测试阶段
% 定义训练-测试阶段数据的组数
data_group=10;
% 定义网络的参数 
initLen=500;           %消除暂态  (washout)
trainLen=6500;         %训练集大小
validationLen=1600;    %验证集大小（用来防止过拟合，训练集上训练之后，在验证集上选择最优参数）
inSize=4;              %输入的序列的维度
outSize=4;             %输出输入的序列的维度
lb  = [1e-6, 0, 1e-6, 1000,0.01,1e-9];                         % 优化参数目标下限（IS,a,SR,k,density,reg）
ub  = [1, 1, 2,2000,0.05,1e-1];                               % 优化参数目标上限（IS,a,SR,k,density,reg）
% 定义优化算法的一些参数    
pop = 10;                                 % 狼的数量
Max_iteration = 150;                      % 最大迭代次数
dim=size(lb,2);                           % 需要优化超参数的个数

%% 开始时间
start_time = datetime('now'); % 记录开始时间
fprintf('开始时间：%s\n', datestr(start_time, 'yyyy-mm-dd HH:MM:SS'));

%%  优化算法寻找最优的超参数组合
[~, Best_pos, curve]=WOA(pop, Max_iteration, lb,ub,dim,data_group);

%% 自己本机的地址
%save('C:\Users\wangjunwen\Desktop\article2\code\MCSs\curve\WOA_curve.mat', 'curve');
%save('C:\Users\wangjunwen\Desktop\article2\code\MCSs\Best_pos\WOA_pos.mat', 'Best_pos');

%% 服务器上的地址
save('/home/user/Desktop/article2/code/MCSs/curve/WOA_curve.mat', 'curve');
save('/home/user/Desktop/article2/code/MCSs/Best_pos/WOA_pos.mat', 'Best_pos');
%% 在后面的指定位置打上结束时间标注
end_time = datetime('now'); % 记录结束时间
fprintf('结束时间：%s\n', datestr(end_time, 'yyyy-mm-dd HH:MM:SS'));

%% 计算并输出花费的时间
elapsed_time = end_time - start_time; % 计算花费的时间
fprintf('花费时间：%s\n', datestr(elapsed_time, 'HH:MM:SS')); % 输出花费的时间
%% 保存时间信息
save('/home/user/Desktop/article2/code/MCSs/time/WOA_time.mat', 'elapsed_time');
%% 获得了最优的参数
disp('经过训练-验证阶段后获得了最好的超参数组合');
disp(Best_pos);
%% 画图
figure(1)
plot(1 : length(curve), curve, 'LineWidth', 1.5);