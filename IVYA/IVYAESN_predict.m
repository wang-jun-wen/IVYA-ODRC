function Pdata=IVYAESN_predict(net, indata, Lbegin, Lend, outSize)
warning('off', 'MATLAB:nearlySingularMatrix');
%%导入训练时的网络参数
a = net.a;
Win = net.Win;
W = net.W;
Wout = net.Wout ;
x = net.x;
b = net.b;

Length=Lend-Lbegin+1;
Y1= zeros(outSize,Length);  
u = indata(Lbegin,:)';   
for t = 1:Length   %测试阶段
    x = (1-a)*x + a*tanh( Win*u + W*x+b );  %得到网络状态
    y = Wout*x;
    Y1(:,t) = y;       %把测试阶段t时刻得到的输出放到Y1中
    % generative mode: 自主预测模式
    u = y;             %t时刻的输入作为下一个时刻的输出
    % this would be a predictive mode:预测模式，用来预测精准度
    %u = indata(Lbegin+t,:)';
end

%输出
Pdata=Y1;
end