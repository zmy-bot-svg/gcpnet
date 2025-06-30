#!/usr/bin/env python3
"""
逐步调试数据集处理过程，在每个关键步骤添加详细日志
"""
import sys
import os
import traceback

# 添加详细的调试信息
def debug_dataset_creation():
    """逐步调试数据集创建过程"""
    print("🔍 逐步调试数据集创建...")
    
    try:
        print("步骤1: 导入必要模块...")
        sys.path.append('.')
        from utils.dataset_utils import MP18
        from torch_geometric.transforms import Compose
        from utils.transforms import GetAngle, ToFloat
        print("✅ 模块导入成功")
        
        print("\n步骤2: 创建变换...")
        transform = Compose([GetAngle(), ToFloat()])
        print("✅ 变换创建成功")
        
        print("\n步骤3: 初始化MP18数据集...")
        print("   参数:")
        print("   - root: './data'")
        print("   - name: 'jarvis_fe_15k'")
        print("   - points: 1")
        print("   - target_name: 'formation_energy_peratom'")
        
        dataset = MP18(
            root='./data', 
            name='jarvis_fe_15k',
            transform=transform,
            r=8.0, 
            n_neighbors=12, 
            edge_steps=50, 
            image_selfloop=True, 
            points=1,  # 只处理1个样本
            target_name='formation_energy_peratom'
        )
        
        print("✅ 数据集初始化成功")
        print(f"✅ 数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            print("\n步骤4: 获取第一个样本...")
            sample = dataset[0]
            print("✅ 样本获取成功")
            print(f"   样本类型: {type(sample)}")
            
            # 检查样本属性
            attrs = ['x', 'edge_index', 'edge_attr', 'y', 'pos', 'z']
            for attr in attrs:
                if hasattr(sample, attr):
                    value = getattr(sample, attr)
                    print(f"   {attr}: {value.shape if hasattr(value, 'shape') else value}")
                else:
                    print(f"   {attr}: 不存在")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 错误发生在: {type(e).__name__}: {str(e)}")
        print("\n详细错误信息:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_dataset_creation()
    if success:
        print("\n🎉 数据集创建调试成功！问题可能在其他地方")
    else:
        print("\n❌ 发现数据集创建问题")