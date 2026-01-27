"""
使用示例脚本
展示如何使用BioTTA进行生物测量和端点定位
"""
import os
import torch
from main import load_config, process_single_image, process_batch, get_image_paths


def example_single_file():
    """示例1: 处理单个文件"""
    print("="*60)
    print("示例1: 处理单个文件")
    print("="*60)
    
    # 加载配置
    config = load_config('config.yaml')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 处理单个文件
    image_path = '/path/to/your/image.nii.gz'
    
    if os.path.exists(image_path):
        result = process_single_image(image_path, config, device)
        print("\n处理结果:")
        print(f"图像名称: {result['image_name']}")
        print(f"预测长度: {result['lengths']}")
        print(f"端点数量: {len(result['landmarks'])}")
    else:
        print(f"文件不存在: {image_path}")


def example_batch_processing():
    """示例2: 批量处理"""
    print("="*60)
    print("示例2: 批量处理")
    print("="*60)
    
    # 加载配置
    config = load_config('config.yaml')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 批量处理文件夹
    input_folder = '/path/to/your/images/'
    
    if os.path.exists(input_folder):
        image_paths = get_image_paths(input_folder)
        print(f"找到 {len(image_paths)} 个图像文件")
        
        results = process_batch(image_paths, config, device)
        print(f"\n成功处理 {len(results)} 个图像")
    else:
        print(f"文件夹不存在: {input_folder}")


def example_custom_config():
    """示例3: 使用自定义配置"""
    print("="*60)
    print("示例3: 使用自定义配置")
    print("="*60)
    
    # 加载配置
    config = load_config('config.yaml')
    
    # 自定义配置
    config['system']['gpu_id'] = '0'  # 使用GPU 0
    config['paths']['output_dir'] = './custom_results'  # 自定义输出目录
    config['step2']['num_epoch'] = 10  # 增加TTA训练轮数
    
    print("已更新配置:")
    print(f"GPU ID: {config['system']['gpu_id']}")
    print(f"输出目录: {config['paths']['output_dir']}")
    print(f"TTA训练轮数: {config['step2']['num_epoch']}")


if __name__ == "__main__":
    # 运行示例（需要替换实际路径）
    print("BioTTA使用示例")
    print("\n注意: 请修改脚本中的路径为实际路径后再运行")
    print("\n取消注释下面的行来运行示例:")
    print("# example_single_file()")
    print("# example_batch_processing()")
    print("# example_custom_config()")

