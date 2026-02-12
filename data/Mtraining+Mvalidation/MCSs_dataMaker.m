% 新系统参数
a = 30;
b = 1;
c = 36;
d = 0.5;
h = 0.003;
dt = 0.01;                 % 时间步长
num_points = 50000;

% 初始点列表
% 示例：均匀分布在[-1, 1]之间
%initial_points = 2 * rand(10, 4) - 1;
%initial_points = 20 * rand(10, 4) - 10;
initial_points = [
    0.560504136642276,  -0.530440173255187,   0.0940177845726899,  0.858771941937460;
   -0.786694459638831,  -0.137172345072911,   0.706062235443787,  -0.165465861831261;
    0.923796161710107,   0.821295188859046,   0.244110262970132,  -0.900691139348516;
   -0.990731551731865,  -0.636305943394295,  -0.298095238215458,   0.805432219830562;
    0.549820929423005,  -0.472394166956020,   0.0264990797341067,  0.889574379443292;
    0.634606441306866,  -0.708922039230566,  -0.196383932496117,  -0.0182718150638401;
    0.737389410727019,  -0.727862882582673,  -0.848066616618316,  -0.0214947231999623;
   -0.200434701802207,   0.159409174731140,  -0.753362130329669,   0.800107692835324;
   -0.480259194298692,   0.0997204036726640, -0.632184423435167,  -0.261506437759570;
    0.600136960448615,  -0.710090403552546,  -0.520094948670195,  -0.777594489412425
];

% 目标目录
output_dir = 'C:\Users\wangjunwen\Desktop\article2\code\MCSs\data\Mtraining+Mvalidation';

% 新系统动力学方程
new_system = @(state) [...
    a * (2 * state(4)^2 * (state(2) - state(1)) + d * state(1));                  % dx1/dt
    b * (2 * state(4)^2 * (state(1) - state(2)) - state(3));                      % dx2/dt
    c * (state(2) - h * state(3));                                                % dx3/dt
    state(2) - state(1) - 0.01 * state(4)                                         % dx4/dt
];

% 四阶龙格-库塔方法积分
for i = 1:size(initial_points, 1)
    data = zeros(num_points, 4);
    state = initial_points(i, :);
    
    % 生成数据
    for j = 1:num_points
        data(j, :) = state;
        k1 = new_system(state);
        k2 = new_system(state + 0.5 * dt * k1');
        k3 = new_system(state + 0.5 * dt * k2');
        k4 = new_system(state + dt * k3');
        state = state + (dt / 6) * (k1' + 2 * k2' + 2 * k3' + k4');
    end
      
    Mtraining = data(1:6500, :);
    Mvalidation = data(6501:end, :);
    
    % 生成文件名并保存数据
    filename = fullfile(output_dir, sprintf('MCSs_train&validate_data%d.mat',i));
    save(filename, 'Mtraining', 'Mvalidation');
end

disp('MCSs系统数据生成并保存到指定目录完成。');