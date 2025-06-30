#!/usr/bin/python
# -*- encoding: utf-8 -*-

# 导入必要的库用于时间处理、操作系统交互和时间计算
import datetime
import os
import time

# 导入PyTorch相关库用于深度学习模型构建和训练
import torch
import wandb  # 用于实验跟踪和可视化
import torch.nn as nn
import torchmetrics  # 用于计算各种评估指标
from torch_geometric.transforms import Compose  # 用于组合多个数据变换

# 导入自定义模块
from model import GCPNet  # GCPNet模型的主要实现
from utils.keras_callbacks import WandbCallback  # Wandb回调函数
from utils.dataset_utils import MP18, dataset_split, get_dataloader  # 数据集处理工具
from utils.flags import Flags  # 配置参数管理
from utils.train_utils import KerasModel, LRScheduler  # 训练工具和学习率调度器
from utils.transforms import GetAngle, ToFloat  # 数据变换工具

# 设置NumExpr库的最大线程数为24，用于加速数值计算
os.environ["NUMEXPR_MAX_THREADS"] = "24"
# 开启调试模式，用于打印调试信息
debug = True 

# 导入日志相关库
import logging
from logging.handlers import RotatingFileHandler

# 配置日志系统，用于记录训练过程中的重要信息
def log_config(log_file='test.log'):
    # 定义日志格式：[时间戳][日志级别]: 消息内容
    LOG_FORMAT = '[%(asctime)s][%(levelname)s]: %(message)s'
    # 设置日志级别为INFO，记录重要的训练信息
    level = logging.INFO
    # 配置基础日志设置
    logging.basicConfig(level=level, format=LOG_FORMAT)
    # 创建文件日志处理器，支持日志轮转（最大2MB，保留3个备份文件）
    log_file_handler = RotatingFileHandler(filename=log_file, maxBytes=2*1024*1024, backupCount=3)
    # 设置日志格式化器
    formatter = logging.Formatter(LOG_FORMAT)
    log_file_handler.setFormatter(formatter)
    # 将文件处理器添加到根日志记录器
    logging.getLogger('').addHandler(log_file_handler)

# 设置随机种子，确保实验的可重复性
def set_seed(seed):
    # 导入随机数生成相关库
    import random
    import numpy as np
    # 设置Python原生random模块的随机种子
    random.seed(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置PyTorch CPU操作的随机种子
    torch.manual_seed(seed)
    # 设置PyTorch GPU操作的随机种子（适用于所有GPU）
    torch.cuda.manual_seed_all(seed)
    # 启用确定性算法，确保GPU计算结果可重复
    torch.backends.cudnn.deterministic = True
    # 禁用cudnn的benchmark模式，虽然可能影响性能但保证结果一致
    # benchmark模式会根据输入数据的大小自动选择最优算法，这可能导致不同运行间的结果不一致
    # 关闭cudnn的benchmark模式，确保每次运行都使用相同的算法
    torch.backends.cudnn.benchmark = False

# 设置和初始化数据集，这是GCPNet训练的第一步
# config: 配置对象，包含数据集相关的参数
def setup_dataset(config):
    # 创建MP18数据集实例，这是一个用于材料性质预测的数据集
    # root: 数据集根目录路径
    # name: 数据集名称
    # transform: 数据变换pipeline，包括获取角度信息和类型转换
    # r: 最大边距离，定义原子间连接的最大距离
    # n_neighbors: 每个原子考虑的最近邻数量
    # edge_steps: 边特征的输入维度
    # image_selfloop: 是否添加自环连接
    # points: 数据点数量
    # target_name: 目标属性名称
    dataset = MP18(root=config.dataset_path, name=config.dataset_name, transform=Compose([GetAngle(), ToFloat(
        )]), r=config.max_edge_distance, n_neighbors=config.n_neighbors, edge_steps=config.edge_input_features, image_selfloop=True, points=config.points, target_name=config.target_name)
    return dataset

# 初始化GCPNet模型，配置模型的各种超参数
def setup_model(dataset, config):
    # 创建GCPNet实例，这是核心的图卷积预测网络
    net = GCPNet(
            data=dataset,  # 传入数据集用于模型初始化
            firstUpdateLayers=config.firstUpdateLayers,  # 第一次更新的层数
            secondUpdateLayers=config.secondUpdateLayers,  # 第二次更新的层数
            atom_input_features=config.atom_input_features,  # 原子特征输入维度
            edge_input_features=config.edge_input_features,  # 边特征输入维度
            triplet_input_features=config.triplet_input_features,  # 三元组特征输入维度
            embedding_features=config.embedding_features,  # 嵌入特征维度
            hidden_features=config.hidden_features,  # 隐藏层特征维度
            output_features=config.output_features,  # 输出特征维度
            min_edge_distance=config.min_edge_distance,  # 最小边距离
            max_edge_distance=config.max_edge_distance,  # 最大边距离
            link=config.link,  # 连接方式配置
            dropout_rate=config.dropout_rate,  # Dropout率，用于防止过拟合
        )
    return net

# 设置优化器，用于模型参数的更新
def setup_optimizer(net, config):
    # 动态获取指定的优化器类（如Adam、SGD等）
    # getattr根据config文件中设定的学习率和其他参数来选择对应的优化器
    optimizer = getattr(torch.optim, config.optimizer)(
        net.parameters(),  # 传入模型的所有可训练参数
        lr=config.lr,  # 学习率
        **config.optimizer_args  # 其他优化器特定参数（如weight_decay等）
    )
    # 如果启用调试模式，打印优化器信息
    if config.debug:
        print(f"optimizer: {optimizer}")
    return optimizer

# 设置学习率调度器，用于在训练过程中动态调整学习率
def setup_schduler(optimizer, config):
    # 创建自定义的学习率调度器实例
    # 可以根据训练进度自动调整学习率，提高训练效果
    scheduler = LRScheduler(optimizer, config.scheduler, config.scheduler_args)
    return scheduler

# 构建Keras风格的模型包装器，简化训练流程
# keras风格的模型封装器，提供统一的接口和功能
def build_keras(net, optimizer, scheduler):
    # 创建KerasModel实例，这是对PyTorch模型的高级封装
    model = KerasModel(
        net=net,  # 传入网络模型
        loss_fn=nn.L1Loss(),  # 损失函数使用L1损失（平均绝对误差）
        metrics_dict={  # 评估指标字典
            "mae": torchmetrics.MeanAbsoluteError(),  # 平均绝对误差
            "mape": torchmetrics.MeanAbsolutePercentageError()  # 平均绝对百分比误差
        }, 
        optimizer=optimizer,  # 优化器
        lr_scheduler=scheduler  # 学习率调度器
    )
    return model

# 主要的训练函数，执行完整的模型训练流程
def train(config, printnet=False):
    # 生成基于当前时间的唯一实验名称，格式：年月日_时分秒
    name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 实验跟踪和超参数管理模块
    # 这部分代码负责初始化实验跟踪系统和处理超参数搜索
    
    # 如果启用了wandb日志记录，初始化Weights & Biases实验跟踪
    if config.log_enable:
        # 初始化wandb项目，用于实验跟踪和可视化
        # project: 项目名称，用于组织相关实验
        # name: 当前实验的唯一名称（基于时间戳）
        # save_code: 设置为False，不自动保存代码到wandb（避免隐私泄露）
        wandb.init(project=config.project_name, name=name, save_code=False)
    
    # 超参数搜索功能：如果任务类型是超参数搜索，从wandb配置中获取超参数
    if config.task_type.lower() == 'hyperparameter':
        # 遍历wandb配置中的所有超参数，依次打印超参数名称和对应的值
        # wandb.config包含了sweep配置文件中定义的搜索空间参数
        # 在超参数搜索过程中，wandb会自动为每次试验分配不同的参数组合
        for k, v in wandb.config.items():
            # 动态更新config对象中的超参数值
            # 使用setattr函数将wandb分配的超参数值覆盖config中的默认值
            # 这样模型就会使用wandb指定的超参数进行训练
            setattr(config, k, v)
            # 打印当前试验使用的超参数，便于调试和监控
            print(f"searched keys: {k}: {v}")

    # 第1步：加载和准备数据
    # 设置数据集
    dataset = setup_dataset(config)
    # 将数据集分割为训练集、验证集和测试集
    # 比例分别为80%、15%、5%
    train_dataset, val_dataset, test_dataset = dataset_split(
        dataset, train_size=0.8, valid_size=0.1, test_size=0.1, seed=config.seed, debug=debug) 
    # 创建数据加载器，用于批量加载数据
    train_loader, val_loader, test_loader = get_dataloader(
        train_dataset, val_dataset, test_dataset, config.batch_size, config.num_workers)

    # 第2步：加载和初始化网络模型
    # 选择计算设备：如果有GPU则使用GPU，否则使用CPU
    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化模型并移动到指定设备（如果有gpu移动到gpu）
    net = setup_model(dataset, config).to(rank)
    # 如果启用调试模式，打印网络结构
    if config.debug:
        print(net)

    # 第3步：设置优化器和学习率调度器
    # 初始化优化器
    optimizer = setup_optimizer(net, config)
    # 初始化学习率调度器
    scheduler = setup_schduler(optimizer, config)

    # 第4步：开始训练过程
    # 如果启用日志记录，设置wandb回调
    if config.log_enable:
        callbacks = [WandbCallback(project=config.project_name,config=config)]
    else:
        callbacks = None
    
    # 构建Keras风格的模型
    model = build_keras(net, optimizer, scheduler)
    # 开始训练模型
    # fit方法执行完整的训练循环，包括：
    # - 前向传播
    # - 损失计算
    # - 反向传播
    # - 参数更新
    # - 验证评估
    # - 早停检查
    model.fit(
        train_loader,  # 训练数据加载器
        val_loader,   # 验证数据加载器
        ckpt_path=os.path.join(config.output_dir, config.net+'.pth'),  # 模型检查点保存路径，+是拼接作用
        epochs=config.epochs,  # 训练轮数
        monitor='val_loss',    # 监控的指标（验证损失）
        mode='min',           # 监控模式（最小化验证损失）
        patience=config.patience,  # 早停耐心值，越大越不容易提前停止
        plot=True,            # 是否绘制训练曲线
        callbacks=callbacks   # 回调函数列表
    )
    # 在测试集上评估模型性能
    print(model.evaluate(test_loader))
    
    # 如果启用日志记录，记录最终的测试结果
    if config.log_enable:
        wandb.log({
            "test_mae":model.evaluate(test_loader)['val_mae'],      # 测试集平均绝对误差
            "test_mape":model.evaluate(test_loader)['val_mape'],    # 测试集平均绝对百分比误差
            "total_params":model.total_params()                     # 模型总参数数量
        })
        # 结束wandb会话
        wandb.finish()

    # 返回训练好的模型
    return model

# 重新导入日志相关库（重复导入，可能是代码整理遗留）
import logging
from logging.handlers import RotatingFileHandler

# 交叉验证训练函数，用于更可靠的模型性能评估
def train_CV(config):

    # 第1步：加载数据并进行交叉验证分割
    # 导入交叉验证相关的数据处理函数
    from utils.dataset_utils_strong import loader_setup_CV, split_data_CV
    # 设置数据集
    dataset = setup_dataset(config)
    # 将数据集分割为多个折叠，用于交叉验证
    cv_dataset = split_data_CV(dataset, num_folds=config.num_folds, seed=config.seed)
    # 初始化存储每个折叠误差的列表
    cv_error = []

    # 遍历每个交叉验证折叠
    for index in range(0, len(cv_dataset)):

        # 为每个折叠设置工作目录
        # 使用海象运算符（:=）同时赋值和检查目录是否存在
        if not(os.path.exists(output_dir := f"{config.output_dir}/{index}")):
            # 如果目录不存在，则创建目录
            os.makedirs(output_dir)

        # 为当前折叠设置数据加载器
        train_loader, test_loader, train_dataset, _ = loader_setup_CV(
            index, config.batch_size, cv_dataset, num_workers=config.num_workers
        )
        
        # 第2步：加载网络模型
        # 选择计算设备
        rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化模型并移动到设备
        net = setup_model(dataset, config).to(rank)
        
        # 第3步：设置优化器和调度器
        optimizer = setup_optimizer(net, config)
        scheduler = setup_schduler(optimizer, config)

        # 第4步：开始当前折叠的训练
        # 构建模型
        model = build_keras(net, optimizer, scheduler)
        # 训练模型（注意：交叉验证中没有单独的验证集，所以val_loader为None）
        model.fit(
            train_loader, 
            None,  # 交叉验证中不使用单独的验证集
            ckpt_path=os.path.join(output_dir, config.net+'.pth'), 
            epochs=config.epochs,
            monitor='train_loss',  # 监控训练损失而不是验证损失
            mode='min', 
            patience=config.patience, 
            plot=True
        )

        # 在测试集上评估当前折叠的性能
        test_error = model.evaluate(test_loader)['val_mae']
        # 记录当前折叠的结果
        logging.info("fold: {:d}, Test Error: {:.5f}".format(index+1, test_error)) 
        # 将误差添加到列表中
        cv_error.append(test_error)
    
    # 计算交叉验证的统计结果
    import numpy as np
    # 计算所有折叠的平均误差
    mean_error = np.array(cv_error).mean()
    # 计算误差的标准差
    std_error = np.array(cv_error).std()
    # 记录最终的交叉验证结果
    logging.info("CV Error: {:.5f}, std Error: {:.5f}".format(mean_error, std_error))
    return cv_error

# 预测函数，用于在新数据上进行推理
def predict(config):
    
    # 第1步：加载数据
    # 设置数据集
    dataset = setup_dataset(config)
    # 导入PyTorch Geometric的数据加载器
    from torch_geometric.loader import DataLoader
    # 创建测试数据加载器（不打乱顺序，用于预测）
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=False,)

    # 第2步：加载网络模型
    # 选择计算设备
    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化模型
    net = setup_model(dataset, config).to(rank)

    # 第3步：设置优化器和调度器（预测时实际不需要，但保持接口一致）
    optimizer = setup_optimizer(net, config)
    scheduler = setup_schduler(optimizer, config)

    # 第4步：开始预测
    # 构建模型
    model = build_keras(net, optimizer, scheduler)
    # 执行预测，加载预训练模型并输出预测结果
    model.predict(test_loader, ckpt_path=config.model_path, test_out_path=config.output_path)

# 可视化函数，用于分析和可视化模型的特征表示
def visualize(config):
    
    # 第1步：加载数据
    # 重新导入必要的模块（局部导入）
    from utils.dataset_utils_strong import MP18, dataset_split, get_dataloader
    from utils.transforms import GetAngle, ToFloat

    # 创建数据集实例
    dataset = MP18(root=config.dataset_path,name=config.dataset_name,transform=Compose([GetAngle(),ToFloat()]), r=config.max_edge_distance, n_neighbors=config.n_neighbors, edge_steps=config.edge_input_features, image_selfloop=True,points=config.points,target_name=config.target_name)

    # 分割数据集（比例与训练时相同）
    train_dataset, val_dataset, test_dataset = dataset_split(dataset,train_size=0.8,valid_size=0.1,test_size=0.1,seed=config.seed, debug=debug)
    # 创建数据加载器
    train_loader, val_loader, test_loader = get_dataloader(train_dataset, val_dataset, test_dataset, config.batch_size, config.num_workers)

    # 第2步：加载网络模型
    # 选择计算设备
    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化模型
    net = setup_model(dataset, config).to(rank)

    # 第3步：设置优化器和调度器
    optimizer = setup_optimizer(net, config)
    scheduler = setup_schduler(optimizer, config)
    # 打印优化器信息（用于调试）
    print("optimizer:",optimizer)

    # 第4步：开始分析
    # 创建模型实例
    model = KerasModel(net=net, loss_fn=nn.L1Loss(), metrics_dict={"mae":torchmetrics.MeanAbsoluteError(),"mape":torchmetrics.MeanAbsolutePercentageError()},optimizer=optimizer,lr_scheduler = scheduler)
    # 准备用于分析的数据加载器
    data_loader, _, _ = get_dataloader(dataset, val_dataset, test_dataset, config.batch_size, config.num_workers)
   
    # 执行模型分析，包括特征可视化（如t-SNE）
    model.analysis(net_name=config.net, test_data=data_loader,ckpt_path=config.model_path,tsne_args=config.visualize_args)

    return model

# 主程序入口点，程序从这里开始执行
if __name__ == "__main__":

    # 忽略PyTorch的TypedStorage废弃警告，为了避免在使用旧版本PyTorch时出现警告信息
    import warnings
    warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')
    
    # 初始化配置管理器，加载所有配置参数
    flags = Flags()
    config = flags.updated_config
    
    # 生成基于时间的唯一输出目录名称
    name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config.output_dir = os.path.join(config.output_dir, name)
    # 创建输出目录（如果不存在）
    if not(os.path.exists(config.output_dir)):
        os.makedirs(config.output_dir)
    # 设置随机种子以确保实验可重复
    set_seed(config.seed)

    # 根据配置的任务类型执行相应的操作
    # GCPNet支持多种运行模式：
    
    # 标准训练模式：训练单个模型
    if config.task_type.lower() == 'train':
        train(config)
    
    # 超参数搜索模式：使用Wandb进行自动超参数优化
    elif config.task_type.lower() == 'hyperparameter':
        # 创建Wandb超参数搜索实例
        sweep_id = wandb.sweep(config.sweep_args, config.entity, config.project_name)
        # 定义搜索过程中的训练函数
        def wandb_train():
            return train(config)
        # 启动超参数搜索代理，执行指定次数的搜索
        wandb.agent(sweep_id, wandb_train, count=config.sweep_count)
    
    # 可视化模式：分析模型的特征表示
    elif config.task_type.lower() == 'visualize':
        visualize(config)
    
    # 交叉验证模式：使用K折交叉验证评估模型
    elif config.task_type.lower() == 'cv':
        # 设置日志文件
        log_file = config.project_name + '.log'
        log_config(log_file)
        # 执行交叉验证训练
        train_CV(config)
    
    # 预测模式：使用训练好的模型进行预测
    elif config.task_type.lower() == 'predict':
        predict(config)
    
    # 如果任务类型不被支持，抛出异常
    else:
        raise NotImplementedError(f"Task type {config.task_type} not implemented. Supported types: train, test, cv, predict")
