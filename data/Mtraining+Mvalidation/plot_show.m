% 设置目标目录
data_dir = 'C:\Users\wangjunwen\Desktop\article2\code\MCSs\data\Mtraining+Mvalidation';

% 文件名前缀和数量
file_prefix = 'MCSs_train&validate_data';
num_files = 10;

% 时间向量
t = (0:1999);

% 自定义柔和的颜色
x_color = [0.1, 0.6, 0.8]; % 蓝绿色
y_color = [0.8, 0.4, 0.1]; % 柔和的橙色
z_color = [0.5, 0.2, 0.6]; % 紫色

% 遍历每个文件并绘制图像
for i = 1:num_files
    % 生成文件名并加载数据
    filename = fullfile(data_dir, sprintf('%s%d.mat', file_prefix, i));
    data = load(filename);
    Mtraining = data.Mtraining;
    %% 
    Mtraining=Mtraining(:,1:3);
    %Mtraining=Mtraining(:,2:4);
    %Mtraining = Mtraining(:, [1, 2, 4]);
    %Mtraining = Mtraining(:, [1, 3, 4]);
    %% 

    % 获取当前数据集的初始点坐标
    initial_point = Mtraining(1, :);
    
    % 创建一个新的图窗口
    figure('Position', [100, 100, 1400, 800]); % 设置窗口大小

    % 绘制左侧的 3D 轨迹图，使用实线
    subplot('Position', [0.05, 0.1, 0.4, 0.8]);
    plot3(Mtraining(:, 1), Mtraining(:, 2), Mtraining(:, 3), 'b', 'LineWidth', 2); % 实线，蓝色，线条加粗
    hold on;
    % 标记并显示当前初始点坐标
    plot3(initial_point(1), initial_point(2), initial_point(3), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    text(initial_point(1), initial_point(2), initial_point(3), ...
        sprintf('(%0.4f, %0.4f, %0.4f)', initial_point(1), initial_point(2), initial_point(3)), ...
        'FontSize', 10, 'Color', 'k', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
    hold off;
    
    grid on;
    title(sprintf('3D图 - %s%d', file_prefix, i), 'Interpreter', 'none');
    xlabel('x');
    ylabel('y');
    zlabel('z');
    view(45, 30); % 设置固定视角
    axis tight;
    box on;

    % 绘制右侧三个单独的维度图，使用柔和的颜色
    subplot('Position', [0.55, 0.7, 0.4, 0.25]);
    plot(t, Mtraining(1:2000, 1), 'Color', x_color, 'LineWidth', 1.5);
    title('3个维度的前200个时间步');
    xlabel('Time (t)');
    ylabel('x');
    grid on;

    subplot('Position', [0.55, 0.4, 0.4, 0.25]);
    plot(t, Mtraining(1:2000, 2), 'Color', y_color, 'LineWidth', 1.5);
    xlabel('Time (t)');
    ylabel('y');
    grid on;

    subplot('Position', [0.55, 0.1, 0.4, 0.25]);
    plot(t, Mtraining(1:2000, 3), 'Color', z_color, 'LineWidth', 1.5);
    xlabel('Time (t)');
    ylabel('z');
    grid on;
end

disp('图像绘制完成，并显示各自的初始点坐标。');
