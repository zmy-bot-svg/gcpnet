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

class JarvisConfig:
    """配置管理类"""
    # 根据实际数据集调整的属性名称
    DEFAULT_ELECTRONIC_BUNDLE = [
        'optb88vdw_bandgap',  # 带隙相关
        'formation_energy_peratom',  # 形成能
        'density',  # 密度
        'nat'  # 原子数
    ]
    
    # 备选的电子性质组合（需要根据实际数据集验证）
    ALTERNATIVE_ELECTRONIC_BUNDLE = [
        'bandgap',
        'bulk_modulus', 
        'shear_modulus',
        'total_energy'
    ]

class JarvisDataHandler:
    """
    一个更通用的JARVIS数据处理器，支持探索和特定任务的数据准备
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
        # 添加数据验证
        if use_cache:
            df = self._load_cache()
            if df is not None:
                # 验证数据完整性
                if len(df) == 0:
                    print("警告：缓存数据为空，重新下载...")
                    return self.download_jarvis_data(use_cache=False)
                return df
        
        # 添加重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"下载尝试 {attempt + 1}/{max_retries}...")
                full_data = data('dft_3d_2021')
                df = pd.DataFrame(full_data)
                
                # 验证下载的数据
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

    def explore_available_properties(self, df):
        """探索数据集中可用的所有属性"""
        print("\n" + "="*60)
        print("数据集中所有可用属性探索")
        print("="*60)
        
        print(f"数据集形状: {df.shape}")
        print(f"总列数: {len(df.columns)}")
        print("\n所有列名:")
        for i, col in enumerate(df.columns):
            print(f"{i+1:3d}. {col}")
        
        # 保存列名到文件
        columns_file = os.path.join(self.cache_dir, 'available_columns.txt')
        with open(columns_file, 'w') as f:
            for col in df.columns:
                f.write(f"{col}\n")
        print(f"\n所有列名已保存到: {columns_file}")
        
        return list(df.columns)

    def find_similar_properties(self, df, target_properties):
        """查找与目标属性相似的实际存在的属性"""
        available_cols = df.columns.tolist()
        similar_props = {}
        
        print("\n" + "="*60)
        print("查找相似属性")
        print("="*60)
        
        for target in target_properties:
            print(f"\n查找与 '{target}' 相似的属性:")
            # 查找包含关键词的列
            similar = []
            
            # 提取关键词
            keywords = []
            if 'bandgap' in target.lower():
                keywords = ['bandgap', 'band', 'gap']
            elif 'dielectric' in target.lower():
                keywords = ['dielectric', 'eps', 'permittivity']
            elif 'mass' in target.lower():
                keywords = ['mass', 'effective']
            elif 'energy' in target.lower():
                keywords = ['energy', 'formation']
            
            # 搜索相似列名
            for col in available_cols:
                col_lower = col.lower()
                for keyword in keywords:
                    if keyword in col_lower:
                        similar.append(col)
                        break
            
            if similar:
                print(f"  找到相似属性: {similar}")
                similar_props[target] = similar
            else:
                print(f"  未找到相似属性")
                similar_props[target] = []
        
        return similar_props

    def validate_property_bundle(self, df, property_bundle):
        """验证属性组合的有效性并提供替代方案"""
        print(f"\n验证属性组合: {property_bundle}")
        
        available_cols = df.columns.tolist()
        valid_props = []
        invalid_props = []
        
        for prop in property_bundle:
            if prop in available_cols:
                valid_props.append(prop)
                print(f"✓ {prop} - 可用")
            else:
                invalid_props.append(prop)
                print(f"✗ {prop} - 不可用")
        
        if invalid_props:
            print(f"\n发现 {len(invalid_props)} 个无效属性，正在查找替代方案...")
            similar_props = self.find_similar_properties(df, invalid_props)
            
            # 提供替代建议
            print("\n建议的替代属性组合:")
            suggested_bundle = valid_props.copy()
            
            for invalid_prop in invalid_props:
                if similar_props[invalid_prop]:
                    # 选择第一个相似属性作为替代
                    suggested = similar_props[invalid_prop][0]
                    suggested_bundle.append(suggested)
                    print(f"  {invalid_prop} -> {suggested}")
            
            return suggested_bundle, valid_props, invalid_props
        else:
            print("✓ 所有属性都可用！")
            return property_bundle, valid_props, []

    def explore_data_completeness(self, df):
        """增强版数据完整性分析"""
        print("\n" + "="*50)
        print("数据完整性探索报告")
        print("="*50)
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        report = []
        
        for col in numeric_cols:
            if col == 'jid': continue
            
            valid_count = df[col].notna().sum()
            valid_ratio = valid_count / len(df) * 100
            
            # 添加基本统计信息
            if valid_count > 0:
                mean_val = df[col].mean()
                std_val = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
            else:
                mean_val = std_val = min_val = max_val = np.nan
            
            report.append({
                'property': col,
                'valid_count': valid_count,
                'valid_ratio': valid_ratio,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val
            })
        
        report_df = pd.DataFrame(report).sort_values(by='valid_ratio', ascending=False)
        print(report_df.to_string())
        
        # 保存报告到文件
        report_file = os.path.join(self.cache_dir, 'data_completeness_report.csv')
        report_df.to_csv(report_file, index=False)
        print(f"\n详细报告已保存到: {report_file}")
        
        return report_df

    def plot_correlation_matrix(self, df, property_bundle, save_plot=True):
        """改进的相关性矩阵绘制"""
        print(f"\n正在为 {len(property_bundle)} 个性质生成相关性矩阵...")
        
        # 验证属性是否存在
        missing_props = [prop for prop in property_bundle if prop not in df.columns]
        if missing_props:
            print(f"警告：以下属性不存在于数据集中: {missing_props}")
            # 只使用存在的属性
            property_bundle = [prop for prop in property_bundle if prop in df.columns]
            if len(property_bundle) < 2:
                print("可用属性少于2个，无法生成相关性矩阵")
                return None
        
        # 只使用有数据的样本计算相关性
        clean_data = df[property_bundle].dropna()
        if len(clean_data) < 10:
            print("警告：可用于相关性分析的样本太少!")
            return None
        
        corr_data = clean_data.corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        
        # 使用更好的颜色方案
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdBu_r', 
                    center=0, fmt='.3f', square=True,
                    cbar_kws={"shrink": .8})
        
        plt.title(f'性质相关性矩阵 (n={len(clean_data)})', fontsize=14)
        plt.tight_layout()
        
        if save_plot:
            plot_file = os.path.join(self.cache_dir, 'correlation_matrix.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"相关性图已保存到: {plot_file}")
        
        plt.show()
        return corr_data

    def suggest_optimal_property_bundle(self, df, min_data_ratio=0.8, max_properties=6):
        """基于数据完整性自动推荐最优属性组合"""
        print(f"\n自动推荐属性组合 (最少数据比例: {min_data_ratio*100}%, 最多属性数: {max_properties})")
        
        # 获取数据完整性报告
        completeness_df = self.explore_data_completeness(df)
        
        # 筛选高完整性的数值属性
        high_quality_props = completeness_df[
            (completeness_df['valid_ratio'] >= min_data_ratio * 100) & 
            (completeness_df['property'] != 'jid')
        ].head(max_properties)
        
        suggested_bundle = high_quality_props['property'].tolist()
        
        print(f"\n推荐的高质量属性组合:")
        for i, prop in enumerate(suggested_bundle, 1):
            ratio = high_quality_props[high_quality_props['property'] == prop]['valid_ratio'].iloc[0]
            print(f"{i}. {prop} (完整度: {ratio:.1f}%)")
        
        return suggested_bundle

    def validate_mtl_dataset(self, df, property_bundle):
        """验证MTL数据集的质量"""
        print("\n数据质量检查:")
        
        # 检查异常值
        for prop in property_bundle:
            if prop in df.columns:
                Q1 = df[prop].quantile(0.25)
                Q3 = df[prop].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[prop] < (Q1 - 1.5 * IQR)) | (df[prop] > (Q3 + 1.5 * IQR))]
                print(f"{prop}: {len(outliers)} 个异常值 ({len(outliers)/len(df)*100:.1f}%)")
        
        # 检查数据分布
        print(f"\n数据集统计摘要:")
        print(df[property_bundle].describe())
        
        return True

    def get_mtl_dataset(self, df, property_bundle):
        """(建模功能) 准备用于多任务学习的最终数据集"""
        print("\n" + "="*50)
        print("准备用于多任务学习 (MTL) 的数据集")
        print("="*50)
        
        # 验证属性组合
        validated_bundle, valid_props, invalid_props = self.validate_property_bundle(df, property_bundle)
        
        if invalid_props:
            print(f"\n使用验证后的属性组合: {validated_bundle}")
            property_bundle = validated_bundle
        
        cols_to_get = ['jid', 'atoms'] + property_bundle
        
        # 确保atoms列存在，如果不存在则跳过
        if 'atoms' not in df.columns:
            print("警告：'atoms' 列不存在，将跳过该列")
            cols_to_get = ['jid'] + property_bundle
        
        # 1. 筛选核心列
        available_cols = [col for col in cols_to_get if col in df.columns]
        mtl_df = df[available_cols].copy()
        print(f"已选择 {len(available_cols)} 个核心列: {available_cols}")
        
        # 2. 关键清洗步骤：只保留在所有目标性质上都有值的样本
        initial_count = len(mtl_df)
        mtl_df.dropna(subset=property_bundle, inplace=True)
        final_count = len(mtl_df)
        print(f"数据清洗 (要求所有目标性质非空): {initial_count} -> {final_count}")
        
        # 3. 数据质量验证
        if final_count > 0:
            self.validate_mtl_dataset(mtl_df, property_bundle)
            
            # 4. 保存清洗后的数据集
            clean_data_file = os.path.join(self.cache_dir, 'mtl_clean_dataset.csv')
            mtl_df.to_csv(clean_data_file, index=False)
            print(f"\n清洗后的数据集已保存到: {clean_data_file}")
        else:
            print("警告：清洗后没有剩余数据！")
        
        return mtl_df.reset_index(drop=True)

def main():
    """主函数示例"""
    # ===================================================================
    # 1. 实例化处理器并获取原始数据
    # ===================================================================
    handler = JarvisDataHandler()
    raw_df = handler.download_jarvis_data(use_cache=True)
    
    if raw_df is None:
        print("无法获取数据，程序终止。")
        return

    # ===================================================================
    # 2. 探索数据集中可用的属性
    # ===================================================================
    print("\n探索数据集结构...")
    available_properties = handler.explore_available_properties(raw_df)
    
    # ===================================================================
    # 3. 自动推荐最优属性组合
    # ===================================================================
    print("\n自动推荐属性组合...")
    suggested_bundle = handler.suggest_optimal_property_bundle(raw_df, min_data_ratio=0.9, max_properties=4)
    
    # ===================================================================
    # 4. 使用推荐的属性组合进行分析
    # ===================================================================
    if suggested_bundle:
        print(f"\n使用推荐的属性组合: {suggested_bundle}")
        
        print("\n生成相关性矩阵...")
        correlation_matrix = handler.plot_correlation_matrix(raw_df, suggested_bundle)
        
        # ===================================================================
        # 5. 获取用于最终建模的、干净的MTL数据集
        # ===================================================================
        final_mtl_df = handler.get_mtl_dataset(raw_df, suggested_bundle)

        print("\n最终数据集预览:")
        print(final_mtl_df.head())
        print(f"\n最终数据集形状: {final_mtl_df.shape}")
    
    # ===================================================================
    # 6. 可选：手动指定属性组合（基于探索结果）
    # ===================================================================
    print("\n" + "="*50)
    print("可选：手动测试其他属性组合")
    print("="*50)
    
    # 基于实际可用的属性创建一个测试组合
    config = JarvisConfig()
    manual_bundle = config.DEFAULT_ELECTRONIC_BUNDLE
    
    print(f"测试手动属性组合: {manual_bundle}")
    validated_bundle, valid_props, invalid_props = handler.validate_property_bundle(raw_df, manual_bundle)
    
    if len(validated_bundle) >= 2:
        print(f"使用验证后的组合: {validated_bundle}")
        manual_corr = handler.plot_correlation_matrix(raw_df, validated_bundle)
        manual_mtl_df = handler.get_mtl_dataset(raw_df, validated_bundle)
        print(f"手动组合最终数据集形状: {manual_mtl_df.shape}")
    
    print("\n处理完成！所有结果文件已保存到 jarvis_cache 目录。")
    
    # 后续，您可以将最终数据集用于您的模型训练流程...

if __name__ == "__main__":
    main()
