# 确保已安装必要的库 
# pip install -U jarvis-tools pandas matplotlib seaborn numpy

from jarvis.db.figshare import data
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OptimizedJarvisConfig:
    """基于实际数据分析优化的配置管理类"""
    
    # 🏆 最终推荐组合：电子性质核心包 + 形成能辅助任务
    FINAL_MTL_BUNDLE = [
        'optb88vdw_bandgap',              # PBE带隙（基础，数据最全）
        'dfpt_piezo_max_dielectric',      # 介电常数（核心电子性质）
        'avg_elec_mass',                  # 电子有效质量
        'avg_hole_mass',                  # 空穴有效质量
        'formation_energy_peratom'        # 形成能（强辅助任务，稳定训练）
    ]
    
    # 备选组合（用于对比实验）
    BASIC_COMPLETE_BUNDLE = [
        'formation_energy_peratom',
        'optb88vdw_bandgap', 
        'optb88vdw_total_energy',
        'density'
    ]
    
    # 纯电子性质组合（高科学价值）
    PURE_ELECTRONIC_BUNDLE = [
        'optb88vdw_bandgap',
        'mbj_bandgap',
        'hse_gap',
        'avg_elec_mass',
        'avg_hole_mass'
    ]

class JarvisDataHandler:
    """
    针对GCPNet-Plus优化的JARVIS数据处理器
    """
    
    def __init__(self, cache_dir='jarvis_cache'):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, 'dft_3d_2021_full.pkl')
        self.metadata_file = os.path.join(cache_dir, 'dft_3d_2021_metadata.pkl')

    def _save_cache(self, df):
        """保存完整数据集的缓存"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(df, f)
            metadata = {'download_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            print(f"原始数据已缓存到 {self.cache_file}")
        except Exception as e:
            print(f"保存缓存失败: {e}")

    def _load_cache(self):
        """加载缓存"""
        if os.path.exists(self.cache_file):
            print(f"从缓存文件 {self.cache_file} 加载...")
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"加载缓存失败: {e}")
        return None

    def download_jarvis_data(self, use_cache=True):
        """下载或从缓存加载完整的dft_3d_2021数据集"""
        if use_cache:
            df = self._load_cache()
            if df is not None:
                if len(df) == 0:
                    print("警告：缓存数据为空，重新下载...")
                    return self.download_jarvis_data(use_cache=False)
                return df
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"下载尝试 {attempt + 1}/{max_retries}...")
                full_data = data('dft_3d_2021')
                df = pd.DataFrame(full_data)
                
                if len(df) == 0:
                    raise ValueError("下载的数据集为空")
                
                self._save_cache(df)
                print(f"✓ 下载成功！共获取 {len(df)} 条材料记录。")
                return df
            except Exception as e:
                print(f"✗ 第 {attempt + 1} 次尝试失败: {e}")
                if attempt == max_retries - 1:
                    print(f"✗ 所有下载尝试失败，请检查网络连接或稍后重试")
                    return None
        return None

    def clean_and_convert_data_types(self, df):
        """清理和转换数据类型"""
        print(f"\n🔧 清理和转换数据类型...")
        
        # 识别应该是数值的列
        potential_numeric_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in [
                'bandgap', 'energy', 'mass', 'density', 'modulus', 'gap',
                'dielectric', 'piezo', 'seebeck', 'kappa', 'formation',
                'total', 'bulk', 'shear', 'poisson', 'ehull'
            ]):
                potential_numeric_cols.append(col)
        
        conversion_report = []
        
        for col in potential_numeric_cols:
            if col in df.columns:
                original_type = str(df[col].dtype)
                original_non_null = df[col].notna().sum()
                
                try:
                    # 尝试转换为数值
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    new_non_null = df[col].notna().sum()
                    converted_successfully = new_non_null > 0
                    
                    conversion_report.append({
                        'column': col,
                        'original_type': original_type,
                        'final_type': str(df[col].dtype),
                        'original_valid': original_non_null,
                        'final_valid': new_non_null,
                        'conversion_success': converted_successfully,
                        'data_loss': original_non_null - new_non_null
                    })
                    
                    if converted_successfully:
                        if new_non_null < original_non_null:
                            print(f"✓ {col}: {original_type} -> {df[col].dtype} (损失 {original_non_null - new_non_null} 个值)")
                        else:
                            print(f"✓ {col}: {original_type} -> {df[col].dtype}")
                    else:
                        print(f"⚠️ {col}: 转换失败，所有值变为NaN")
                        
                except Exception as e:
                    print(f"✗ {col}: 转换失败 - {e}")
                    conversion_report.append({
                        'column': col,
                        'original_type': original_type,
                        'final_type': original_type,
                        'original_valid': original_non_null,
                        'final_valid': original_non_null,
                        'conversion_success': False,
                        'data_loss': 0
                    })
        
        # 保存转换报告
        if conversion_report:
            report_df = pd.DataFrame(conversion_report)
            report_file = os.path.join(self.cache_dir, 'data_type_conversion_report.csv')
            report_df.to_csv(report_file, index=False)
            print(f"\n数据类型转换报告已保存到: {report_file}")
        
        return df

    def explore_column_types(self, df):
        """探索所有列的数据类型和内容"""
        print(f"\n🔍 探索数据列类型和内容...")
        
        type_analysis = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null_count = df[col].notna().sum()
            null_count = df[col].isna().sum()
            
            # 检查数据内容示例
            sample_values = df[col].dropna().head(3).tolist()
            sample_str = str(sample_values)[:50] + "..." if len(str(sample_values)) > 50 else str(sample_values)
            
            type_analysis.append({
                'column': col,
                'dtype': dtype,
                'non_null_count': non_null_count,
                'null_count': null_count,
                'sample_values': sample_str
            })
        
        analysis_df = pd.DataFrame(type_analysis)
        
        # 保存类型分析
        type_file = os.path.join(self.cache_dir, 'column_type_analysis.csv')
        analysis_df.to_csv(type_file, index=False)
        print(f"列类型分析已保存到: {type_file}")
        
        # 显示潜在的问题列
        print(f"\n⚠️ 需要特别注意的列:")
        problematic_cols = analysis_df[analysis_df['dtype'] == 'object']
        for _, row in problematic_cols.iterrows():
            print(f"  {row['column']} (object类型): {row['sample_values']}")
        
        return analysis_df

    def safe_numeric_stats(self, series):
        """安全地计算数值统计信息"""
        try:
            if pd.api.types.is_numeric_dtype(series):
                valid_data = series.dropna()
                if len(valid_data) > 0:
                    return {
                        'mean': float(valid_data.mean()),
                        'std': float(valid_data.std()),
                        'min': float(valid_data.min()),
                        'max': float(valid_data.max())
                    }
            return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        except Exception as e:
            print(f"计算统计信息时出错: {e}")
            return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}

    def validate_property_bundle(self, df, property_bundle, bundle_name=""):
        """验证属性组合的有效性并提供详细统计"""
        print(f"\n{'='*60}")
        print(f"验证属性组合{bundle_name}: {property_bundle}")
        print(f"{'='*60}")
        
        available_cols = df.columns.tolist()
        validation_results = []
        
        for prop in property_bundle:
            if prop in available_cols:
                valid_count = df[prop].notna().sum()
                valid_ratio = valid_count / len(df) * 100
                
                # 安全地计算统计信息
                stats = self.safe_numeric_stats(df[prop])
                
                # 修复格式化问题：将dtype转换为字符串
                dtype_str = str(df[prop].dtype)
                
                validation_results.append({
                    'property': prop,
                    'status': '✓ 可用',
                    'dtype': dtype_str,
                    'valid_count': valid_count,
                    'valid_ratio': f'{valid_ratio:.1f}%',
                    'mean': f'{stats["mean"]:.4f}' if not np.isnan(stats["mean"]) else 'N/A',
                    'std': f'{stats["std"]:.4f}' if not np.isnan(stats["std"]) else 'N/A'
                })
                
                # 检查数据类型 - 修复格式化问题
                if pd.api.types.is_numeric_dtype(df[prop]):
                    print(f"✓ {prop:<35} | 类型: {dtype_str:<10} | 有效数据: {valid_count:>6} ({valid_ratio:>5.1f}%)")
                else:
                    print(f"⚠️ {prop:<35} | 类型: {dtype_str:<10} | 有效数据: {valid_count:>6} ({valid_ratio:>5.1f}%) [非数值类型]")
                    
            else:
                validation_results.append({
                    'property': prop,
                    'status': '✗ 不可用',
                    'dtype': 'N/A',
                    'valid_count': 0,
                    'valid_ratio': '0.0%',
                    'mean': 'N/A',
                    'std': 'N/A'
                })
                print(f"✗ {prop:<35} | 不存在于数据集中")
        
        # 保存验证结果
        results_df = pd.DataFrame(validation_results)
        validation_file = os.path.join(self.cache_dir, f'validation_{bundle_name.lower().replace(" ", "_")}.csv')
        results_df.to_csv(validation_file, index=False)
        print(f"\n验证结果已保存到: {validation_file}")
        
        valid_props = [r['property'] for r in validation_results if '✓' in r['status']]
        invalid_props = [r['property'] for r in validation_results if '✗' in r['status']]
        
        # 检查数值类型的有效属性
        numeric_valid_props = []
        for prop in valid_props:
            if prop in df.columns and pd.api.types.is_numeric_dtype(df[prop]):
                numeric_valid_props.append(prop)
        
        if len(numeric_valid_props) != len(valid_props):
            print(f"\n⚠️ 注意: {len(valid_props) - len(numeric_valid_props)} 个属性不是数值类型，将在后续分析中跳过")
        
        return numeric_valid_props, invalid_props, validation_results

    def analyze_data_intersection(self, df, property_bundle):
        """分析多个性质的数据交集，这对MTL至关重要"""
        print(f"\n分析 {len(property_bundle)} 个性质的数据交集:")
        
        # 只使用数值类型的列
        numeric_props = [prop for prop in property_bundle 
                        if prop in df.columns and pd.api.types.is_numeric_dtype(df[prop])]
        
        if len(numeric_props) != len(property_bundle):
            excluded = set(property_bundle) - set(numeric_props)
            print(f"⚠️ 排除非数值属性: {excluded}")
        
        # 逐步分析数据交集
        intersection_analysis = []
        current_data = df.copy()
        
        for i, prop in enumerate(numeric_props):
            before_count = len(current_data)
            current_data = current_data.dropna(subset=[prop])
            after_count = len(current_data)
            lost_count = before_count - after_count
            
            intersection_analysis.append({
                'step': i + 1,
                'added_property': prop,
                'remaining_samples': after_count,
                'lost_samples': lost_count,
                'loss_rate': f'{lost_count/before_count*100:.1f}%' if before_count > 0 else '0.0%'
            })
            
            print(f"Step {i+1}: 加入 {prop}")
            print(f"  剩余样本: {after_count:,} (损失: {lost_count:,}, {lost_count/before_count*100:.1f}%)")
        
        # 保存交集分析结果
        intersection_df = pd.DataFrame(intersection_analysis)
        intersection_file = os.path.join(self.cache_dir, 'data_intersection_analysis.csv')
        intersection_df.to_csv(intersection_file, index=False)
        print(f"\n数据交集分析已保存到: {intersection_file}")
        
        final_count = len(current_data)
        original_count = len(df)
        print(f"\n📊 最终统计:")
        print(f"   原始数据集: {original_count:,} 样本")
        print(f"   交集数据集: {final_count:,} 样本")
        print(f"   保留率: {final_count/original_count*100:.1f}%")
        
        return current_data, intersection_analysis, numeric_props

    def plot_enhanced_correlation_matrix(self, df, property_bundle, bundle_name="", save_plot=True):
        """绘制增强版相关性矩阵"""
        print(f"\n为 {bundle_name} 生成相关性矩阵...")
        
        # 确保所有属性都是数值类型
        numeric_props = [prop for prop in property_bundle 
                        if prop in df.columns and pd.api.types.is_numeric_dtype(df[prop])]
        
        if len(numeric_props) != len(property_bundle):
            excluded = set(property_bundle) - set(numeric_props)
            print(f"⚠️ 排除非数值属性: {excluded}")
        
        if len(numeric_props) < 2:
            print(f"❌ 数值属性少于2个，无法生成相关性矩阵")
            return None, None
        
        # 只使用有完整数据的样本
        clean_data = df[numeric_props].dropna()
        if len(clean_data) < 100:
            print(f"警告：可用样本过少 (n={len(clean_data)})，建议至少100个样本")
            if len(clean_data) < 10:
                return None, None
        
        corr_data = clean_data.corr()
        
        # 创建更美观的图表
        plt.figure(figsize=(14, 12))
        
        # 创建mask只显示下三角
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        
        # 绘制热力图
        sns.heatmap(corr_data, 
                    mask=mask, 
                    annot=True, 
                    cmap='RdBu_r', 
                    center=0, 
                    fmt='.3f', 
                    square=True,
                    cbar_kws={"shrink": .8, "label": "Pearson Correlation"},
                    annot_kws={"size": 12})
        
        plt.title(f'{bundle_name} 性质相关性矩阵\n(样本数: {len(clean_data):,})', 
                  fontsize=16, pad=20)
        plt.xlabel('性质', fontsize=14)
        plt.ylabel('性质', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_plot:
            plot_file = os.path.join(self.cache_dir, f'correlation_matrix_{bundle_name.lower().replace(" ", "_")}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"相关性图已保存到: {plot_file}")
        
        plt.show()
        
        # 输出关键相关性统计
        print(f"\n相关性统计摘要:")
        correlations = []
        for i in range(len(numeric_props)):
            for j in range(i+1, len(numeric_props)):
                corr_val = corr_data.iloc[i, j]
                correlations.append({
                    'property_1': numeric_props[i],
                    'property_2': numeric_props[j],
                    'correlation': corr_val,
                    'abs_correlation': abs(corr_val)
                })
        
        correlations_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
        print("最强的性质间相关性 (前5个):")
        for _, row in correlations_df.head().iterrows():
            print(f"  {row['property_1']} ↔ {row['property_2']}: {row['correlation']:.3f}")
        
        return corr_data, correlations_df

    def create_final_mtl_dataset(self, df, property_bundle, dataset_name="final_mtl"):
        """创建最终的MTL数据集"""
        print(f"\n{'='*60}")
        print(f"创建最终MTL数据集: {dataset_name}")
        print(f"{'='*60}")
        
        # 验证属性
        valid_props, invalid_props, _ = self.validate_property_bundle(df, property_bundle, dataset_name)
        
        if invalid_props:
            print(f"\n⚠️  发现无效属性: {invalid_props}")
            print(f"继续使用有效属性: {valid_props}")
            property_bundle = valid_props
        
        if len(property_bundle) < 2:
            print("❌ 有效属性少于2个，无法创建MTL数据集")
            return None, None
        
        # 分析数据交集
        clean_data, intersection_analysis, final_props = self.analyze_data_intersection(df, property_bundle)
        
        if len(clean_data) == 0:
            print("❌ 数据交集为空，无法创建数据集")
            return None, None
        
        # 构建最终数据集
        essential_cols = ['jid']
        if 'atoms' in df.columns:
            essential_cols.append('atoms')
        
        final_cols = essential_cols + final_props
        final_dataset = clean_data[final_cols].copy().reset_index(drop=True)
        
        # 数据质量检查
        print(f"\n📋 最终数据集质量报告:")
        print(f"   样本数量: {len(final_dataset):,}")
        print(f"   特征数量: {len(final_props)}")
        print(f"   总列数: {len(final_cols)}")
        
        # 异常值检测
        print(f"\n🔍 异常值检测 (使用IQR方法):")
        for prop in final_props:
            if pd.api.types.is_numeric_dtype(final_dataset[prop]):
                Q1 = final_dataset[prop].quantile(0.25)
                Q3 = final_dataset[prop].quantile(0.75)
                IQR = Q3 - Q1
                outliers = final_dataset[
                    (final_dataset[prop] < (Q1 - 1.5 * IQR)) | 
                    (final_dataset[prop] > (Q3 + 1.5 * IQR))
                ]
                outlier_ratio = len(outliers) / len(final_dataset) * 100
                print(f"   {prop}: {len(outliers)} 个异常值 ({outlier_ratio:.1f}%)")
        
        # 保存数据集和元数据
        dataset_file = os.path.join(self.cache_dir, f'{dataset_name}_dataset.csv')
        final_dataset.to_csv(dataset_file, index=False)
        
        # 保存元数据
        metadata = {
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_samples': len(df),
            'final_samples': len(final_dataset),
            'retention_rate': len(final_dataset) / len(df) * 100,
            'properties': final_props,
            'dataset_name': dataset_name
        }
        
        metadata_file = os.path.join(self.cache_dir, f'{dataset_name}_metadata.json')
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 文件已保存:")
        print(f"   数据集: {dataset_file}")
        print(f"   元数据: {metadata_file}")
        
        return final_dataset, metadata

def main():
    """
    主函数：实现完整的数据选择和准备流程
    这是为GCPNet-Plus项目量身定制的最终版本
    """
    print("🚀 GCPNet-Plus 数据处理流程启动")
    print("="*60)
    
    # ===================================================================
    # 步骤 1: 数据加载和预处理
    # ===================================================================
    handler = JarvisDataHandler()
    raw_df = handler.download_jarvis_data(use_cache=True)
    
    if raw_df is None:
        print("❌ 无法获取数据，程序终止")
        return
    
    # 探索数据类型
    handler.explore_column_types(raw_df)
    
    # 清理和转换数据类型
    raw_df = handler.clean_and_convert_data_types(raw_df)
    
    config = OptimizedJarvisConfig()
    
    # ===================================================================
    # 步骤 2: 🏆 使用最终推荐的属性组合 
    # ===================================================================
    print(f"\n🎯 使用最终推荐的MTL属性组合...")
    
    FINAL_MTL_BUNDLE = config.FINAL_MTL_BUNDLE
    print(f"最终组合包含 {len(FINAL_MTL_BUNDLE)} 个性质:")
    for i, prop in enumerate(FINAL_MTL_BUNDLE, 1):
        print(f"  {i}. {prop}")
    
    # 验证最终组合
    valid_props, invalid_props, _ = handler.validate_property_bundle(
        raw_df, FINAL_MTL_BUNDLE, "最终MTL组合"
    )
    
    if len(valid_props) >= 2:  # 至少需要2个有效属性
        # 生成相关性矩阵
        corr_matrix, corr_stats = handler.plot_enhanced_correlation_matrix(
            raw_df, valid_props, "最终MTL组合"
        )
        
        # 创建最终数据集
        final_dataset, metadata = handler.create_final_mtl_dataset(
            raw_df, valid_props, "gcpnet_plus_final"
        )
        
        if final_dataset is not None:
            print(f"\n🎉 最终数据集创建成功!")
            print(f"   数据集形状: {final_dataset.shape}")
            print(f"   可用于训练GCPNet-Plus模型")
            
            # 显示数据预览
            print(f"\n📋 数据集预览:")
            print(final_dataset.head())
            
            # 只显示数值列的统计信息
            numeric_cols = [col for col in final_dataset.columns if pd.api.types.is_numeric_dtype(final_dataset[col])]
            if numeric_cols:
                print(f"\n📊 目标性质统计:")
                print(final_dataset[numeric_cols].describe())
    
    # ===================================================================
    # 步骤 3: 对比实验 - 分析其他组合
    # ===================================================================
    print(f"\n🔬 对比分析：其他属性组合的表现")
    print("="*60)
    
    # 分析基础完整组合
    basic_bundle = config.BASIC_COMPLETE_BUNDLE
    print(f"\n📌 分析基础完整组合 (用于对比):")
    valid_basic, _, _ = handler.validate_property_bundle(
        raw_df, basic_bundle, "基础完整组合"
    )
    
    if len(valid_basic) >= 2:
        basic_corr, _ = handler.plot_enhanced_correlation_matrix(
            raw_df, valid_basic, "基础完整组合"
        )
        basic_dataset, _ = handler.create_final_mtl_dataset(
            raw_df, valid_basic, "basic_complete"
        )
        print(f"基础组合最终数据集形状: {basic_dataset.shape if basic_dataset is not None else 'Failed'}")
    
    # ===================================================================
    # 步骤 4: 总结和建议
    # ===================================================================
    print(f"\n🏁 处理完成 - 总结报告")
    print("="*60)
    print(f"✅ 所有分析结果已保存到 '{handler.cache_dir}' 目录")
    print(f"✅ 主要输出文件:")
    print(f"   - gcpnet_plus_final_dataset.csv (🏆 推荐用于GCPNet-Plus训练)")
    print(f"   - gcpnet_plus_final_metadata.json (数据集元信息)")
    print(f"   - correlation_matrix_*.png (各组合的相关性图)")
    print(f"   - column_type_analysis.csv (数据类型分析)")
    print(f"   - data_type_conversion_report.csv (类型转换报告)")
    
    print(f"\n🚀 下一步建议:")
    print(f"   1. 检查数据类型转换报告，确认关键属性转换成功")
    print(f"   2. 使用 'gcpnet_plus_final_dataset.csv' 训练您的模型")
    print(f"   3. 基于相关性分析结果优化模型架构")
    print(f"   4. 考虑将数据集拆分为训练/验证/测试集")

if __name__ == "__main__":
    main()
