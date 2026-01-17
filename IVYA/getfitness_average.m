function mean_mae = getfitness_average(hyperpara_set, data_group)
    % 初始化误差存储数组
    mae_set = zeros(data_group, 1);
    warning('off', 'MATLAB:nearlySingularMatrix');
    % 使用parfor进行并行计算
    parfor i = 1:data_group
        % 动态加载每组数据并存储在输出变量中
        %data_path = sprintf('C:\\Users\\wangjunwen\\Desktop\\article2\\code\\MCSs\\data\\Mtraining+Mvalidation\\MCSs_train&validate_data%d.mat', i);
        data_path = sprintf('//home//user//Desktop//article2//code//MCSs//dataMtraining+Mvalidation//MCSs_train&validate_data%d.mat', i);
        data = load(data_path, 'Mtraining', 'Mvalidation');  % 使用带输出的load
        
        % 从结构体中提取数据
        Mtraining = data.Mtraining;
        Mvalidation = data.Mvalidation;

        % 拼接训练和验证数据集
        indata = [Mtraining; Mvalidation];   % 拼接输入数据
        outdata = [Mtraining; Mvalidation];  % 拼接输出数据

        % 设置随机种子以保证每个并行任务的独立性
        rng(i, 'twister');

        % 使用hyperpara_set中提供的超参数进行适应度计算
        mae = getfitness(indata, outdata, hyperpara_set);
        
        % 将计算得到的RMSE存储到数组中
        mae_set(i) = mae;
    end
    % 计算平均RMSE作为返回值
    mean_mae= mean(mae_set);

    % 输出平均RMSE
    %fprintf('\nMean RMSE is %f\n', mean_rmse);
end

