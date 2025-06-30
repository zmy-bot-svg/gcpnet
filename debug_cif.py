# 文件名: debug_hang.py
import os
import torch
from utils.flags import Flags
from utils.dataset_utils import MP18
from utils.transforms import GetAngle, ToFloat
from torch_geometric.transforms import Compose

def run_debug():
    print("🔧 GCPNet 全面问题诊断", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载配置
    print("\n--- [诊断] 正在加载 config.yml ---", flush=True)
    try:
        flags = Flags()
        config = flags.updated_config
        print(f"--- [诊断] 配置加载成功: 将测试数据集 '{config.dataset_name}' ---", flush=True)
    except Exception as e:
        print(f"❌ [诊断] 致命错误：无法加载或解析 config.yml! 错误: {e}", flush=True)
        return

    # 2. 清理环境 (非常重要)
    processed_dir = os.path.join(config.dataset_path, config.dataset_name, 'processed')
    if os.path.exists(processed_dir):
        import shutil
        print(f"--- [诊断] 发现旧的缓存目录，正在删除: {processed_dir} ---", flush=True)
        shutil.rmtree(processed_dir)
        print("--- [诊断] 缓存已删除 ---", flush=True)

    # 3. 尝试创建数据集对象
    # 这一步会触发 dataset_utils.py 中的 process() 方法，我们将在其中看到详细的调试输出
    print("\n--- [诊断] 准备创建 MP18 数据集对象 (这将触发详细的 process 流程)... ---", flush=True)
    try:
        # 这里的 transform 组合与 main.py 中保持一致
        transform = Compose([GetAngle(), ToFloat()])
        
        dataset = MP18(
            root=config.dataset_path,
            name=config.dataset_name,
            transform=transform,
            r=config.max_edge_distance,
            n_neighbors=config.n_neighbors,
            edge_steps=config.edge_input_features,
            points=config.points,
            target_name=config.target_name
        )
        print("\n" + "="*60, flush=True)
        print(f"✅ [诊断] 恭喜！数据集成功创建，共 {len(dataset)} 个样本。", flush=True)
        print("✅ 这意味着您的数据和代码现在是兼容的，可以直接运行 main.py 进行训练了。", flush=True)

    except Exception as e:
        print("\n" + "="*60, flush=True)
        print(f"❌ [诊断] 在数据集创建过程中发生错误！", flush=True)
        print("❌ 请查看上面的 [DEBUG] 日志，最后一条成功打印的日志就是程序卡住或报错的位置。", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_debug()