% 新系统参数
a = 30;
b = 1;
c = 36;
d = 0.5;
h = 0.003;
dt = 0.01;                 % 时间步长
num_points = 500000;

% 初始点列表
initial_points = [
             0.1	-0.1	-0.1	0.1     % 初始状态可根据需要调整
];

% 目标目录
output_dir = 'C:\Users\wangjunwen\Desktop\article2\code\MCSs\data\Mtraining+Mtest';

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
    Mtest = data(6501:end, :);
    
    % 生成文件名并保存数据
    filename = fullfile(output_dir, sprintf('MCSs_train&test_data_3d.mat'));
    save(filename, 'Mtraining', 'Mtest');
end

disp('MCSs系统数据生成并保存到指定目录完成。');