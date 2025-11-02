#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置生成脚本
根据 config.yaml 生成多种参数组合的配置文件
"""

import os
import yaml
import itertools
from pathlib import Path
from typing import Dict, List, Any, Optional


class ConfigGenerator:
    """配置生成器类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置生成器
        
        Args:
            config_path: 原始配置文件路径
        """
        self.config_path = config_path
        self.base_config: Dict[str, Any] = {}
        self.output_base_dir = "args"
        
    def load_config(self) -> Dict[str, Any]:
        """
        加载原始配置文件
        
        Returns:
            配置字典
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if config is None:
                raise ValueError("配置文件为空")
            self.base_config = config
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件未找到: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件解析错误: {e}")
    
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        验证参数值的有效性
        
        Args:
            params: 参数字典
            
        Returns:
            是否有效
        """
        # 验证 seed 必须是正整数或0
        if 'seed' in params:
            seed = params['seed']
            if not isinstance(seed, int) or seed < 0:
                return False
        
        # 验证 eval_only 必须是布尔值
        if 'eval_only' in params:
            if not isinstance(params['eval_only'], bool):
                return False
        
        # 验证 trainer 不能为空
        if 'trainer' in params:
            if not params['trainer'] or not isinstance(params['trainer'], str):
                return False
        
        # 验证配置文件路径不为空
        if 'config_file' in params:
            if not params['config_file'] or not isinstance(params['config_file'], str):
                return False
        
        if 'dataset_config_file' in params:
            if not params['dataset_config_file'] or not isinstance(params['dataset_config_file'], str):
                return False
        
        # 验证 NUM_SHOTS 必须是有效值
        if 'DATASET.NUM_SHOTS' in params:
            num_shots = params['DATASET.NUM_SHOTS']
            if not isinstance(num_shots, int) or num_shots <= 0:
                return False
        
        # 验证 SUBSAMPLE_CLASSES 必须是有效值
        if 'DATASET.SUBSAMPLE_CLASSES' in params:
            subsample = params['DATASET.SUBSAMPLE_CLASSES']
            if not isinstance(subsample, str) or subsample not in ['all', 'base', 'new']:
                return False
        
        return True
    
    def generate_combinations(
        self, 
        options: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        生成所有参数组合
        
        Args:
            options: 参数选项字典
            
        Returns:
            参数组合列表
        """
        # 获取所有参数的键和值列表
        param_names = list(options.keys())
        param_values = [options[name] for name in param_names]
        
        # 生成所有组合
        all_combinations = list(itertools.product(*param_values))
        
        # 转换为字典列表
        combinations = []
        for combo in all_combinations:
            combo_dict = dict(zip(param_names, combo))
            
            # 验证参数组合
            if self.validate_parameters(combo_dict):
                combinations.append(combo_dict)
        
        return combinations
    
    def generate_output_dir(self, params: Dict[str, Any]) -> str:
        """
        根据参数生成输出目录路径
        
        Args:
            params: 参数字典
            
        Returns:
            输出目录路径（使用正斜杠，符合YAML规范）
        """
        trainer = params.get('trainer', 'unknown')
        dataset_config = params.get('dataset_config_file', 'unknown')
        config_file = params.get('config_file', 'unknown')
        
        # 移除路径中的目录分隔符和文件扩展名，用于构建路径
        dataset_name = Path(dataset_config).stem  # 获取不带扩展名的文件名
        config_name = Path(config_file).stem
        
        # 构建输出目录（使用正斜杠，符合YAML规范）
        output_dir = os.path.join(
            'output',
            trainer,
            dataset_name,
            config_name,
            self._generate_config_filename_base(params)
        )
        
        return output_dir
    
    def _generate_config_filename_base(self, params: Dict[str, Any]) -> str:
        """
        生成配置文件名的基础部分（不含扩展名）
        
        Args:
            params: 参数字典
            
        Returns:
            文件名基础部分
        """
        trainer = params.get('trainer', 'unknown')
        num_shots = params.get('DATASET.NUM_SHOTS', 0)
        subsample = params.get('DATASET.SUBSAMPLE_CLASSES', 'all')
        eval_only = params.get('eval_only', False)
        seed = params.get('seed', 0)
        
        mode = 'eval' if eval_only else 'train'
        
        filename = f"{trainer}_{num_shots}_{subsample}_{mode}_{seed}"
        
        return filename
    
    def generate_filename(self, params: Dict[str, Any]) -> str:
        """
        生成配置文件名
        
        Args:
            params: 参数字典
            
        Returns:
            文件名
        """
        return self._generate_config_filename_base(params) + ".yaml"
    
    def _build_config_file_path(self, filename: str, trainer: str) -> str:
        """
        根据文件名构建完整的配置文件路径
        
        Args:
            filename: 配置文件名（不含路径，可含或不含扩展名）
            trainer: 训练器名称
            
        Returns:
            完整配置文件路径
        """
        # 如果已经包含路径分隔符，直接返回
        if os.sep in filename or '/' in filename:
            return filename
        
        # 确保文件名有 .yaml 扩展名
        if not filename.endswith('.yaml'):
            filename = filename + '.yaml'
        
        # 构建完整路径
        return f"configs/trainers/{trainer}/{filename}"
    
    def _build_dataset_config_file_path(self, filename: str) -> str:
        """
        根据文件名构建完整的数据集配置文件路径
        
        Args:
            filename: 数据集配置文件名（不含路径，可含或不含扩展名）
            
        Returns:
            完整配置文件路径
        """
        # 如果已经包含路径分隔符，直接返回
        if os.sep in filename or '/' in filename:
            return filename
        
        # 确保文件名有 .yaml 扩展名
        if not filename.endswith('.yaml'):
            filename = filename + '.yaml'
        
        # 构建完整路径
        return f"configs/datasets/{filename}"
    
    def create_config_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据参数组合创建配置文件内容
        
        Args:
            params: 参数字典
            
        Returns:
            配置字典
        """
        # 复制基础配置
        config = self.base_config.copy()
        
        # 获取训练器名称
        trainer = params.get('trainer', config.get('trainer', 'CoOp'))
        
        # 更新参数
        config['seed'] = params.get('seed', config.get('seed', 1))
        config['trainer'] = trainer
        config['eval_only'] = params.get('eval_only', config.get('eval_only', False))
        
        # 构建配置文件路径（如果传入的是文件名，自动添加路径前缀）
        config_file = params.get('config_file', config.get('config_file', ''))
        config['config_file'] = self._build_config_file_path(config_file, trainer)
        
        # 构建数据集配置文件路径（如果传入的是文件名，自动添加路径前缀）
        dataset_config_file = params.get('dataset_config_file', config.get('dataset_config_file', ''))
        config['dataset_config_file'] = self._build_dataset_config_file_path(dataset_config_file)
        
        # 生成并设置 output_dir
        config['output_dir'] = self.generate_output_dir(params)
        
        # 处理 opts 参数
        if 'opts' not in config:
            config['opts'] = []
        
        # 更新或添加 DATASET.NUM_SHOTS
        num_shots = params.get('DATASET.NUM_SHOTS')
        if num_shots is not None:
            self._update_opts(config, 'DATASET.NUM_SHOTS', num_shots)
        
        # 更新或添加 DATASET.SUBSAMPLE_CLASSES
        subsample = params.get('DATASET.SUBSAMPLE_CLASSES')
        if subsample is not None:
            self._update_opts(config, 'DATASET.SUBSAMPLE_CLASSES', subsample)
        
        return config
    
    def _update_opts(self, config: Dict[str, Any], key: str, value: Any):
        """
        更新配置中的 opts 列表
        
        Args:
            config: 配置字典
            key: 选项键
            value: 选项值
        """
        if 'opts' not in config:
            config['opts'] = []
        
        opts = config['opts']
        
        # 查找并更新现有选项
        found = False
        for i in range(len(opts) - 1):
            if opts[i] == key:
                opts[i + 1] = value
                found = True
                break
        
        # 如果未找到，添加新选项
        if not found:
            config['opts'].extend([key, value])
    
    def generate_output_path(self, params: Dict[str, Any]) -> Path:
        """
        生成输出文件路径
        
        Args:
            params: 参数字典
            
        Returns:
            输出文件路径
        """
        trainer = params.get('trainer', 'unknown')
        dataset_config = params.get('dataset_config_file', 'unknown')
        config_file = params.get('config_file', 'unknown')
        
        # 提取文件名（不含扩展名），无论传入的是完整路径还是文件名
        dataset_path = Path(dataset_config)
        config_path = Path(config_file)
        
        # 构建输出目录结构: args/trainer/dataset_config_file/config_file/
        output_dir = Path(self.output_base_dir) / trainer / dataset_path.stem / config_path.stem
        
        # 生成文件名
        filename = self.generate_filename(params)
        
        return output_dir / filename
    
    def save_config(self, config: Dict[str, Any], output_path: Path) -> bool:
        """
        保存配置文件
        
        Args:
            config: 配置字典
            output_path: 输出文件路径
            
        Returns:
            是否成功
        """
        try:
            # 创建目录
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 写入配置文件
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            return True
        except Exception as e:
            print(f"保存配置文件失败 {output_path}: {e}")
            return False
    
    def generate_all_configs(
        self,
        options: Dict[str, List[Any]],
        verbose: bool = True
    ) -> List[Path]:
        """
        生成所有配置文件
        
        Args:
            options: 参数选项字典，键为参数名，值为可选值列表
            verbose: 是否显示详细信息
            
        Returns:
            生成的配置文件路径列表
        """
        # 加载基础配置
        self.load_config()
        
        # 生成所有组合
        combinations = self.generate_combinations(options)
        
        if verbose:
            print(f"共生成 {len(combinations)} 个配置组合")
        
        # 生成配置文件
        generated_files = []
        failed_files = []
        
        for idx, params in enumerate(combinations):
            try:
                # 创建配置内容
                config = self.create_config_content(params)
                
                # 生成输出路径
                output_path = self.generate_output_path(params)
                
                # 保存配置文件
                if self.save_config(config, output_path):
                    generated_files.append(output_path)
                    if verbose and (idx + 1) % 10 == 0:
                        print(f"已生成 {idx + 1}/{len(combinations)} 个配置文件...")
                else:
                    failed_files.append(output_path)
                    
            except Exception as e:
                print(f"处理配置组合失败 (索引 {idx}): {e}")
                failed_files.append(idx)
        
        if verbose:
            print(f"\n生成完成!")
            print(f"成功: {len(generated_files)} 个")
            print(f"失败: {len(failed_files)} 个")
            if failed_files:
                print(f"失败的配置: {failed_files}")
        
        return generated_files


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成配置文件组合')
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
    config_path = os.path.join(base_dir, 'config.yaml')

    parser.add_argument(
        '--config',
        type=str,
        default=config_path,
        help='原始配置文件路径 (默认: config.yaml)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='静默模式（不显示详细信息）'
    )
    
    args = parser.parse_args()
    
    try:
        # 定义参数可选值
        # 用户可以在这里修改参数组合
        # 注意：config_file 和 dataset_config_file 只需提供文件名（不含路径），脚本会自动添加路径前缀
        options = {
            'seed': [1, 2, 3],  # 随机种子
            'trainer': ['CoOp'],  # 训练器名称
            'eval_only': [False, True],  # 评估模式
            'config_file': [
                'vit_b16',
                'vit_b32',
            ],  # 方法配置文件名（不含路径）
            'dataset_config_file': [
                'oxford_pets',
            ],  # 数据集配置文件名（不含路径）
            'DATASET.NUM_SHOTS': [8, 16],  # 样本数量
            'DATASET.SUBSAMPLE_CLASSES': ['all', 'base', 'new'],  # 子采样类别
        }
        
        # 创建生成器
        generator = ConfigGenerator(config_path=args.config)
        
        # 生成所有配置
        generated_files = generator.generate_all_configs(
            options=options,
            verbose=not args.quiet
        )
        
        if not args.quiet:
            print(f"\n所有配置文件已保存到 '{generator.output_base_dir}' 目录下")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

