"""
BioTTA主入口脚本
整合Step1（长度预测）和Step2（端点定位）和Step3（可视化）的完整流程
"""
import os
import sys
import argparse
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import torch
from natsort import natsorted

# 导入模块
from biotta_step1 import predict_lengths
from biotta_step2 import run_tta_and_predict
from biotta_output import save_results, format_landmarks_for_display


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_image_paths(input_path: str) -> List[str]:
    """
    获取图像路径列表：输入路径（文件或文件夹），返回图像路径列表
    """
    input_path = Path(input_path)
    
    if input_path.is_file():
        # 单个文件
        if input_path.suffix in ['.nii', '.gz']:
            return [str(input_path)]
        else:
            raise ValueError(f"不支持的文件格式: {input_path.suffix}")
    elif input_path.is_dir():
        # 文件夹
        image_paths = []
        for ext in ['*.nii.gz', '*.nii']:
            image_paths.extend(list(input_path.glob(ext)))
        image_paths = natsorted([str(p) for p in image_paths])
        return image_paths
    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")


def get_age_for_image(image_path: str, age_file: Optional[str] = None, 
                     age_format: Optional[Dict] = None) -> Optional[float]:
    """
    获取图像的年龄信息：输入年龄文件路径、年龄文件格式配置、图像路径，返回年龄值
    """
    if age_file is None or not os.path.exists(age_file):
        return None
    
    if age_format is None:
        age_format = {'name_column': 'name', 'age_column': 'age'}
    
    try:
        # 尝试读取年龄文件
        if age_file.endswith('.csv'):
            age_df = pd.read_csv(age_file, dtype={age_format['name_column']: str})
        elif age_file.endswith('.tsv'):
            age_df = pd.read_csv(age_file, sep='\t', dtype={age_format['name_column']: str})
        else:
            return None
        
        # 获取图像名称
        image_name = Path(image_path).stem
        if image_name.endswith('.nii'):
            image_name = image_name[:-4]
        
        # 匹配年龄
        name_col = age_format['name_column']
        age_col = age_format['age_column']
        
        print(name_col, age_col)
        if name_col in age_df.columns and age_col in age_df.columns:
            matches = age_df[age_df[name_col] == image_name]
            if len(matches) > 0:
                age = matches.iloc[0][age_col]
                return float(age)
    except Exception as e:
        print(f"⚠ 读取年龄文件失败: {e}")
    
    return None


def process_single_image(
    image_path: str,
    config: Dict,
    device: torch.device,
    age: Optional[float] = None
) -> Dict:
    """
    处理单个图像
    
    Args:
        image_path: 图像路径
        config: 配置字典
        device: 计算设备
        age: 年龄（可选）
    
    Returns:
        结果字典
    """
    print(f"\n{'='*60}")
    print(f"处理图像: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # Step1: 预测长度
    print("\n[Step1] 预测生物测量长度...")
    step1_config = config['step1']
    step1_model_path = config['paths']['step1_model']
    biometry_list = config['data']['biometry_list']
    length_weights = config['data']['length_weights']
    ventricle_indices = [0, 1, 2, 3]
    
    try:
        pred_lengths_df = predict_lengths(
            [image_path],
            step1_model_path,
            device,
            batch_size=step1_config['batch_size'],
            biometry_list=biometry_list,
            length_weights=length_weights,
            ventricle_indices=ventricle_indices
        )
        
        pred_lengths = pred_lengths_df.iloc[0]
        print("✓ Step1 完成")
        print(f"预测的长度值:")
        for i, biometry in enumerate(biometry_list):
            print(f"  {biometry}: {pred_lengths[biometry]:.2f}")
        
        # 保存Step1结果（如果配置允许）
        save_intermediate = config['system'].get('save_intermediate', {})
        if save_intermediate.get('step1_results', True):
            output_dir = config['paths']['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            step1_csv_path = os.path.join(output_dir, 'step1_lengths.csv')
            # 如果文件已存在，追加模式；否则新建
            if os.path.exists(step1_csv_path):
                existing_df = pd.read_csv(step1_csv_path)
                pred_lengths_df = pd.concat([existing_df, pred_lengths_df], ignore_index=True)
            pred_lengths_df.to_csv(step1_csv_path, index=False)
            print(f"✓ Step1结果已保存到: {step1_csv_path}")
    except Exception as e:
        print(f"✗ Step1 失败: {e}")
        raise
    
    # Step2: TTA端点定位
    print("\n[Step2] TTA端点定位...")
    step2_config = config['step2']
    step2_model_path = config['paths']['step2_model']
    
    # 获取年龄（如果未提供）
    if age is None:
        age_file = config['paths'].get('age_file', '')
        age_format = config['data'].get('age_file_format', {})
        age = get_age_for_image(image_path, age_file if age_file else None, age_format)
    
    # 模板路径
    template_paths = None
    if 'template' in config['paths']:
        template_config = config['paths']['template']
        if os.path.exists(template_config.get('image_folder', '')):
            template_paths = {
                'image_folder': template_config['image_folder'],
                'label_folder': template_config['label_folder'],
                'registered_folder': template_config.get('registered_folder', '')
            }
    
    # 是否使用已有TTA模型
    use_existing_tta = step2_config.get('use_existing_tta', False)
    skip_tta = step2_config.get('skip_tta', False)
    
    tta_model_path = None
    if use_existing_tta:
        # 尝试找到对应的TTA模型
        image_name = Path(image_path).stem
        if image_name.endswith('.nii'):
            image_name = image_name[:-4]
        
        output_dir = config['paths']['output_dir']
        tta_model_path = os.path.join(output_dir, image_name, 'tent_TTA.pth')
        if not os.path.exists(tta_model_path):
            tta_model_path = None
    
    # 准备输出目录和中间结果保存配置
    output_dir = config['paths']['output_dir']
    curve_dir = config['paths']['development_curves']
    image_name = Path(image_path).stem.replace('.nii', '')
    per_image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(per_image_output_dir, exist_ok=True)
    save_intermediate = config['system'].get('save_intermediate', {})
    
    try:
        landmarks, metadata = run_tta_and_predict(
            image_path,
            pred_lengths[biometry_list],
            step2_model_path,
            device,
            step2_config,
            template_paths=template_paths,
            age=age,
            skip_tta=skip_tta,
            tta_model_path=tta_model_path if use_existing_tta else None,
            output_dir=per_image_output_dir,
            curve_dir=curve_dir,
            save_intermediate=save_intermediate
        )
        
        print("✓ Step2 完成")
        print(f"预测的端点数量: {len(landmarks)}")
        print("\n前3个端点坐标:")
        for i in range(min(3, len(landmarks))):
            print(f"  Point {i+1}: ({landmarks[i][0]:.2f}, {landmarks[i][1]:.2f}, {landmarks[i][2]:.2f})")
    except Exception as e:
        print(f"✗ Step2 失败: {e}")
        raise
    
    # 构建结果
    result = {
        'image_name': Path(image_path).stem.replace('.nii', ''),
        'age': age,
        'lengths': pred_lengths[biometry_list].to_dict(),
        'landmarks': landmarks
    }
    
    return result


def process_batch(
    image_paths: List[str],
    config: Dict,
    device: torch.device
) -> List[Dict]:
    """
    批量处理图像
    
    Args:
        image_paths: 图像路径列表
        config: 配置字典
        device: 计算设备
    
    Returns:
        结果列表
    """
    print(f"\n开始批量处理 {len(image_paths)} 个图像...\n")
    
    results = []
    
    # 加载年龄文件（如果有）
    age_file = config['paths'].get('age_file', '')
    age_format = config['data'].get('age_file_format', {})
    age_df = None
    
    if age_file and os.path.exists(age_file):
        try:
            if age_file.endswith('.csv'):
                age_df = pd.read_csv(age_file, dtype={age_format.get('name_column', 'name'): str})
            elif age_file.endswith('.tsv'):
                age_df = pd.read_csv(age_file, sep='\t', dtype={age_format.get('name_column', 'name'): str})
        except Exception as e:
            print(f"⚠ 无法加载年龄文件: {e}")
    
    # 处理每个图像
    for idx, image_path in enumerate(image_paths):
        print(f"\n[{idx+1}/{len(image_paths)}]")
        
        # 尝试从年龄文件中获取年龄
        age = None
        if age_df is not None:
            image_name = Path(image_path).stem.replace('.nii', '')
            name_col = age_format.get('name_column', 'name')
            age_col = age_format.get('age_column', 'age')
            
            if name_col in age_df.columns and age_col in age_df.columns:
                matches = age_df[age_df[name_col] == image_name]
                if len(matches) > 0:
                    age = float(matches.iloc[0][age_col])
        
        try:
            result = process_single_image(image_path, config, device, age)
            results.append(result)
        except Exception as e:
            print(f"✗ 处理失败: {e}")
            # 可以选择继续或停止
            continue
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='BioTTA: 胎儿生物测量和端点定位工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例用法:
            # 单个文件（指定年龄）
            python main.py --input path/to/image.nii.gz --age 31 --output ./results
    
            # 文件夹（从CSV读取年龄）
            python main.py --input path/to/folder --age_csv path/to/age.csv --output ./results
        """
    )
    
    # 输入从命令行或者配置文件中获取，在这部分进行第一步处理

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入图像文件或文件夹路径'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='配置文件路径 (默认: config.yaml)'
    )
    
    parser.add_argument(
        '--gpu',
        type=str,
        default=None,
        help='GPU ID (默认: 使用配置文件中的设置)'
    )
    
    parser.add_argument(
        '--age',
        type=int,
        default=None,
        help='年龄值（整数，用于单个文件输入时直接指定）'
    )

    parser.add_argument(
        '--age_csv',
        type=str,
        default=None,
        help='年龄CSV文件路径（用于文件夹批量处理，覆盖config中的设置）。CSV需包含name和age两列'
    )
    
    parser.add_argument(
        '--age_file_format',
        type=str,
        default=None,
        help='覆盖 age_file_format 配置，传入 JSON 字符串，例如: \'{"name_column": "name", "age_column": "age"}\''
    )
    
    parser.add_argument(
        '--registered_folder',
        type=str,
        default=None,
        help='覆盖 registered_folder 配置，传入 JSON 字符串，例如: \'{"name_column": "name", "age_column": "age"}\''
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出目录路径 (默认: 使用配置文件中的设置)'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    if not os.path.exists(args.config):
        print(f"✗ 配置文件不存在: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)

    # 如果预测文件夹并提供了胎龄csv
    if args.age_csv is not None:
        if not os.path.exists(args.age_csv):
            print(f"✗ 年龄CSV文件不存在: {args.age_csv}")
            sys.exit(1)
        config['paths']['age_file'] = args.age_csv
        print(f"使用命令行指定的年龄文件: {args.age_csv}")
    
    # 如果提供了胎龄csv的列名age_file_format
    if args.age_file_format is not None:
        try:
            age_format_dict = json.loads(args.age_file_format)
        except json.JSONDecodeError as e:
            print(f"✗ 解析 --age_file_format 失败: {e}")
            sys.exit(1)
        config['data']['age_file_format'].update(age_format_dict)
    
    # 如果提供了反配准后图谱的保存路径    
    if args.registered_folder is not None:
        config['paths']['template']['registered_folder'] = args.registered_folder
    
    # 设置GPU
    gpu_id = args.gpu or config['system']['gpu_id']
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置输出目录
    if args.output:
        config['paths']['output_dir'] = args.output
    
    # 获取输入路径
    input_path = args.input
    if not input_path:
        input_path = config['paths'].get('input_data', '')
    
    if not input_path:
        print("✗ 未指定输入路径，请使用 --input 参数或配置文件中设置")
        sys.exit(1)
    
    try:
        image_paths = get_image_paths(input_path)
        print(f"找到 {len(image_paths)} 个图像文件")
    except Exception as e:
        print(f"✗ 获取图像路径失败: {e}")
        sys.exit(1)
    
    if len(image_paths) == 0:
        print("✗ 未找到图像文件")
        sys.exit(1)
    
    # 处理图像
    if len(image_paths) == 1:
        # 单文件处理，如果提供了--age参数则使用
        age = args.age
        result = process_single_image(image_paths[0], config, device, age=age)
        results = [result]
    else:
        # 批量处理
        if args.age is not None:
            print("⚠ 批量处理时忽略--age参数（仅对单个文件有效）")
        results = process_batch(image_paths, config, device)
    
    # 保存结果
    if len(results) > 0:
        print(f"\n{'='*60}")
        print("保存结果...")
        print(f"{'='*60}")
        
        output_dir = config['paths']['output_dir']
        output_format = config['output'].get('format', 'json')
        biometry_list = config['data']['biometry_list']
        
        save_results(results, output_dir, output_format, biometry_list)
        
        print(f"\n✓ 处理完成! 共处理 {len(results)} 个图像")
        print(f"结果已保存到: {output_dir}")
    else:
        print("\n✗ 没有成功处理任何图像")


if __name__ == "__main__":
    main()

