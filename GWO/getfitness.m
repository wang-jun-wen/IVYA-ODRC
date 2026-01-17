%fitness函数只用来在寻寻最优参数时求解适应度值
function objValue = getfitness(indata,outdata,Best_pos)
%参数 
initLen=500;           %消除暂态  (washout)
trainLen=6500;         %训练集大小
validationLen=1600;    %验证集大小（用来防止过拟合，训练集上训练之后，在验证集上选择最优参数）
inSize=4;              %输入的序列的维度
outSize=4;             %输出输入的序列的维度

%% ESN参数
IS=Best_pos(1,1);          %Win的放缩比例
a=Best_pos(1,2);           %泄漏率，用于控制储备池中的状态向量的更新速度
SR=Best_pos(1,3);          %谱半径
resSize=Best_pos(1,4);     %储备池邻接矩阵的规模（储备池中节点的个数）
resSize= round(resSize);   %resize必须为正数
density=Best_pos(1,5);     %density
reg=Best_pos(1,6);         %reg
b=Best_pos(1,7);           %输入偏置σb
W=sprand(resSize, resSize, density);              %水库网络的邻接矩阵
Win = (rand(resSize,inSize) * 2 - 1) * IS;   %Win为[-IS,IS]之间的均匀分布
%%训练集训练
net=GWOESN_train(indata, outdata, initLen, trainLen,reg,a,SR,b,resSize,W,Win);

%%验证集上预测
Pdata= GWOESN_predict(net, indata, trainLen, trainLen+validationLen-1, outSize);

%性能评价 计算MAE
accuracy = sum(sum(abs(outdata(trainLen+1:trainLen+validationLen,:) - Pdata'))) /(outSize * validationLen);
objValue = accuracy;
end




