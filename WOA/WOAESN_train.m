function net=WOAESN_train(indata, outdata, initLen, trainLen,reg,a,SR,resSize,W,Win)
warning('off', 'MATLAB:nearlySingularMatrix');
%放缩谱半径
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt)); 
W = W .* (SR/rhoW); %谱半径设置为SR
b=1*ones(resSize,1);
X = zeros(resSize,trainLen-initLen);    %504*2500 一列代表一个个体及其状态叠加。2500个个体
Yt = outdata(initLen+2:trainLen+1,:)';           %3*2500 一列代表一个个体。2500个个体
x = zeros(resSize,1);    %500*1 x代表此时水库网络的状态 一共500个节点
for t = 1:trainLen       %训练开始
    u = indata(t,:)';    %u为输入 就是所谓的一步预测
    x = (1-a)*x + a*tanh( Win*u + W*x+b ); %得到t时刻的状态 
    if t > initLen       %已经消除暂态
        X(:,t-initLen) = x;   %将1，输入，网络状态矩阵竖着拼。作为一项加到X中
    end
end
X_T = X';
Wout = Yt*X_T / (X*X_T + reg*eye(resSize));

%% 赋值 封装网络参数，之后预测阶段不用输入过多参数
net.a = a;
net.Win = Win;
net.W = W;
net.Wout = Wout;
net.x  =  x;
net.b  =  b;
end