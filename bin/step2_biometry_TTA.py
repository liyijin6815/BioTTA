import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from natsort import natsorted
import nibabel as nb
from matplotlib import gridspec
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR

# parent_path = Path(__file__).parent.parent.parent
# sys.path.append(str(parent_path))
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from step2_train_network import * # 导入网络
from supp_v3 import * # 导入数据处理、网络拼装、损失函数、图谱配准、提取坐标模块 #####


# 输入路径与路径处理
# source_model_path = '/data/birth/lmx/work/Class_projects/lyj/work/fetal_localization_SFUDA/checkpoints/source_HLA_only_singlepoint_2/MinValLoss.pth'
# target_image_folder_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/FeTA_dataset/registered/test_2/'
# target_pred_length_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/FeTA_dataset/registered/lyj_measure_2/test_results_v1.csv' #####
# target_age_file_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/FeTA_dataset/participants.tsv'
# template_image_folder_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/template/template_image/'
# template_label_folder_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/template/template_label/'
# template_label_register_to_FeTA_folder_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/template/template_label/register_to_FeTA_SyN/' #####
# target_results_folder_path = '/data/birth/lmx/work/Class_projects/lyj/work/fetal_localization_SFUDA/test_and_results/target_data_TTA_with_template/test_5epoch' #####
source_model_path = '/data/birth/lmx/work/Class_projects/lyj/work/fetal_localization_SFUDA/checkpoints/source_HLA_only_singlepoint_2/MinValLoss.pth'
target_image_folder_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/WC_BTFE_dataset/registered/test/' #####
target_pred_length_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/WC_BTFE_dataset/registered/test_hxt_measure/test_results_v1.csv' #####
target_age_file_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/WC_BTFE_dataset/btfe_age.csv' #####
template_image_folder_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/template/template_image/'
template_label_folder_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/template/template_label/'
template_label_register_to_FeTA_folder_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/template/template_label/register_to_WCB_SyN/' #####
target_results_folder_path = '/data/birth/lmx/work/Class_projects/lyj/work/fetal_localization_SFUDA/final_experiments/WCB/UNet_TTA_with_template/raw_results' #####
# source_model_path = '/data/birth/lmx/work/Class_projects/lyj/work/fetal_localization_SFUDA/checkpoints/source_HLA_only_singlepoint_2/MinValLoss.pth'
# target_image_folder_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/LFC_dataset/registered/test/' #####
# target_pred_length_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/LFC_dataset/registered/test_json_measure/test_results_v1.csv' #####
# target_age_file_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/LFC_dataset/age_all.csv' #####
# template_image_folder_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/template/template_image/'
# template_label_folder_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/template/template_label/'
# template_label_register_to_FeTA_folder_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/template/template_label/register_to_LFC_SyN/' #####
# target_results_folder_path = '/data/birth/lmx/work/Class_projects/lyj/work/fetal_localization_SFUDA/final_experiments/LFC/UNet_TTA_with_template/raw_results' #####
source_model_path = '/data/birth/lmx/work/Class_projects/lyj/work/fetal_localization_SFUDA/checkpoints/source_HLA_only_singlepoint_2/MinValLoss.pth'
target_image_folder_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/WCUMS_dataset/WCT_VM_test/' #####
target_pred_length_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/WCUMS_dataset/WCT_VM_test_Jia_measure/test_results_v1.csv' #####
target_age_file_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/WCUMS_dataset/data_age_week.csv' #####
template_image_folder_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/template/template_image/'
template_label_folder_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/template/template_label/'
template_label_register_to_FeTA_folder_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/template/template_label/register_to_WCT-VM_SyN/' #####
target_results_folder_path = '/data/birth/lmx/work/Class_projects/lyj/work/fetal_localization_SFUDA/final_experiments/WCT-VM/UNet_TTA_with_template/raw_results' #####

if not os.path.exists(template_label_register_to_FeTA_folder_path):
    os.mkdir(template_label_register_to_FeTA_folder_path)
if not os.path.exists(target_results_folder_path):
    os.mkdir(target_results_folder_path)
# 读取age文件
# age_df = pd.read_csv(target_age_file_path, sep='\t')
age_df = pd.read_csv(target_age_file_path, dtype={'name': str})
# 辅助模型预测的长度伪标签文件设置
pred_length_df = pd.read_csv(target_pred_length_path, dtype={'label_name': str}).iloc[:, :-1]


# 输入参数
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
batch_size = 1
channel = 22
init_temp = 0.5
topk = 50 #####
learning_rate = 0.0005 #####
num_epoch = 5 #####
lambda_entropy_loss = 0.5 #####
lambda_length_loss = 1.5 #####
lambda_boundarygrad_loss = 300 #####
point_pairs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
radius = [20, 20, 20, 20, 20, 20, 5, 5, 5, 5, 5] #####
template_label_scale = 1 #####
template_label_pan = 0.25 #####
radius_test = False #####


# 网络初始设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not radius_test:
    net = LocalAppearance(1, channel).to(device) 
    # 加载模型
    pretrained_model = torch.load(source_model_path, map_location=device)  # 加载 .pth 文件，并映射到当前设备
    net.load_state_dict(pretrained_model['net'], strict=False)  # 加载权重，忽略不匹配的键
    print('成功加载包含 "net" 键的预训练权重')
    # 修改网络封装
    tent_net = TENTWrapper(net)  # 封装原始网


# 处理测试集文件夹下的每个文件
test_iamge_list = natsorted(os.listdir(target_image_folder_path))
for file in test_iamge_list:
    print(file)
    target_image_name = file[:-7]

    # 对于一个目标域测试样本设置保存路径
    target_image_path = os.path.join(target_image_folder_path, file)
    target_results_path = os.path.join(target_results_folder_path, target_image_name)
    if not os.path.exists(target_results_path):  
        os.mkdir(target_results_path)
        
    # 跳过已处理的文件 #####
    # if os.path.exists(os.path.join(target_results_path, 'tent_TTA.pth')) \
    # and os.path.exists(os.path.join(target_results_path, 'landmarks.txt')) \
    # and os.path.exists(os.path.join(target_results_path, 'landmarks.png')):
    #     print('已存在结果，跳过！')
    #     continue

    # 测试数据设置
    dataset_target = TestDataset(target_image_path, pred_length_df) 
    num_data_target = len(dataset_target)
    num_batch_target = np.ceil(num_data_target / batch_size)
    target_loader = DataLoader(dataset_target, batch_size=batch_size, shuffle=True, num_workers=22,drop_last=True,pin_memory=True) # dataloader是pytorch一个类，pin_memory加速数据从CPU导入GPU，num_workers设置工作进程数

    # 如果首次进行TTA
    if not radius_test:
        # 损失设置
        detector = DifferentiableLandmarkDetector(init_temp=init_temp, topk=topk) # 创建可微分地标检测器，每张热图提取1个点
        entropy_loss_f = EntropyLoss()  
        length_loss_f = LengthLoss()
        boundarygrad_loss_f = BoundaryGradientLoss(
            point_pairs=point_pairs,
            kernel_type='scharr',    # 使用sobel/scharr算子
            smooth_sigma=1.0,        # 高斯平滑强度
            grad_sigma=0.9,          # 梯度核平滑
            adaptive_threshold=False, # 使用自适应阈值
            min_grad=0.4
            )
        
         # TTA训练设置 
        torch.manual_seed(1) # 设置随机数生成器种子，确定性算法生成伪随机数
        # optim = torch.optim.Adam(tent_net.parameters(), lr=learning_rate) #####
        optim = torch.optim.Adam(
            list(tent_net.parameters()) + list(length_loss_f.parameters()), 
            lr=learning_rate
        )
        scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim,T_max=num_epoch) # 动态调整学习率，在整个num_epoch的周期中学习率先下降再上升，cos
        min_loss = 1000000

        # TENT微调流程
        tent_net.train()  # 保持训练模式以使用当前批次统计量
        log=np.zeros([num_epoch,3])
        best_num = 0
        for epoch in range(num_epoch):
            loss_log = []
            entropy_loss_log = []
            length_loss_log = []
            for i, batch_data in enumerate(target_loader):
                image = batch_data['image'].to(device) # torch.Size([1, 1, 128, 160, 128])
                length_pseudo_label = batch_data['length'].to(device) # torch.Size([1, 11])
                # 前向传播
                _, HLA = tent_net(image) # torch.Size([1, 22, 128, 160, 128])
                # 计算坐标位置
                all_keypoints = []
                for c in range(channel):
                    channel_heatmap = HLA[:, c:c+1, :, :, :] # 获取当前通道热图 [B, 1, D, H, W]
                    keypoints = detector(channel_heatmap) # 使用检测器提取关键点 [B, 1, 3]
                    all_keypoints.append(keypoints.squeeze(1)) # 移除通道维度 [B, 3]
                keypoints = torch.stack(all_keypoints, dim=1) # 组合所有关键点 [B, 22, 3]
                # 计算损失
                entropy_loss = entropy_loss_f(HLA)
                length_loss = length_loss_f(keypoints, length_pseudo_label)
                boundarygrad_loss = boundarygrad_loss_f(image, keypoints)
                loss = lambda_entropy_loss * entropy_loss + lambda_length_loss * length_loss + lambda_boundarygrad_loss * boundarygrad_loss
                # 反向传播
                optim.zero_grad()
                loss.backward()
                # for name, param in tent_net.model.named_parameters():
                #     if '2.weight' in name: 
                #         print('here!')
                #         print(f"{name} 的梯度:", param.grad)  # 应非 None
                # print("LengthLoss参数梯度示例:", length_loss_f.number_scale_factor.grad)
                optim.step()
                # 打印日志
                print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f} = {lambda_entropy_loss:.1f} * {entropy_loss.item():.4f} + {lambda_length_loss:.1f} * {length_loss.item():.4f} + {lambda_boundarygrad_loss:.1f} * {boundarygrad_loss.item():.4f}')
                loss_log += [loss.item()]
                entropy_loss_log += [entropy_loss.item()]
                length_loss_log += [length_loss.item()]
            # 保存最优模型
            if np.mean(loss_log)<min_loss:
                best_num = epoch+1
                min_loss=np.mean(loss_log)
                torch.save({'net': tent_net.state_dict(), 'optim' : optim.state_dict()},"%s/tent_TTA.pth" % (target_results_path))
            # # 记录日志结果
            # log[epoch-1,0]=np.mean(loss_log)
            # log[epoch-1,1]=np.mean(entropy_loss_log)
            # log[epoch-1,2]=np.mean(length_loss_log)
            # log_path = os.path.join(target_results_path, 'TTA_log.npy')
            # np.save(log_path, log)

        # 预测结果
        print('best epoch:', best_num)
        target_model_path = os.path.join(target_results_path, 'tent_TTA.pth')
        TTA_model = torch.load(target_model_path, map_location=device)
        tent_net.load_state_dict(TTA_model['net'], strict=False)
        # tent_net.eval() 其实不加也没影响：因为都是单样本数据不存在批次分布差异；也不会梯度回传
        _, HLA = tent_net(image)
        heatmap = HLA.detach().cpu().numpy().astype(np.float32).squeeze()
    
    else:

        # 直接导入已经过TTA的模型
        target_TTA_model_path = os.path.join(target_results_folder_path, target_image_name, 'tent_TTA.pth')
        TTA_model = torch.load(target_TTA_model_path, map_location=device)
        net = LocalAppearance(1, channel).to(device) 
        tent_net = TENTWrapper(net)
        tent_net.load_state_dict(TTA_model['net'], strict=False)  # 加载权重，忽略不匹配的键
        print('已完成TTA，成功加载包含 "net" 键的预训练权重并修改网络封装')

        # 测试结果 
        for i, batch_data in enumerate(target_loader): # 其实batcn = 1
            image = batch_data['image'].to(device) # torch.Size([1, 1, 128, 160, 128])
            length_pseudo_label = batch_data['length'].to(device) # torch.Size([1, 11])
            _, HLA = tent_net(image) # torch.Size([1, 11, 128, 160, 128])
            heatmap = HLA.detach().cpu().numpy().astype(np.float32).squeeze()


    # 保存output热图结果
    img_affine = nb.load(target_image_path).affine  
    # for j in range(1,23): 
    #     output_file_path = os.path.join(target_results_path, f'TTA_heatmap_{j}.nii.gz')
    #     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    #     output_nifti_img = nb.Nifti1Image(heatmap[j-1], img_affine)
    #     output_nifti_img.to_filename(output_file_path)


    # 再次导入图片用于图谱配准先验约束和画图
    data = nb.load(target_image_path).get_fdata()
    data_copy = data.copy()
    data_expand = np.expand_dims(data_copy, -1)
    mask = data_expand > 0 # 提取mask
    ind_brain = block_ind(mask) # 提取mask区域索引
    sized_data = extract_brain(data,ind_brain,[128,160,128]) # 大小缩放
    sized_mask = np.zeros([128,160,128])
    sized_mask = sized_data > 0 # mask大小缩放
    max_value = np.max(sized_data)
    min_value = np.min(sized_data) 


    # 图谱配准并保存反配准的模板热图
    # age = age_df.loc[age_df['participant_id'] == target_image_name, 'Gestational age'].values[0]
    # age = age_df.loc[age_df['image_name'] == target_image_name, 'age'].values[0]
    age = age_df.loc[age_df['name'] == target_image_name, 'week_float'].values[0]
    age = round(age)
    print(age)
    if age <= 22:
        print("年龄小于23岁，没有对应图谱热图")
    else:
        # 判断是否已经存在配准结果
        template_label_register_to_FeTA_file_path = os.path.join(template_label_register_to_FeTA_folder_path, f'{target_image_name}_registered_{age}_m.nii.gz')
        if not os.path.exists(template_label_register_to_FeTA_file_path): # 如果不存在文件则进行配准
            template_image_path = os.path.join(template_image_folder_path, f'STA{age}.nii.gz')
            template_label_path = os.path.join(template_label_folder_path, f'{age}_m.nii.gz')
            output_path = template_label_register_to_FeTA_file_path
            # 配准模板热图并保存
            register_template_label(template_image_path, template_label_path, target_image_path, output_path)
            print(f"经过配准生成热图，年龄{age}")            
        else: # 如果已存在文件无需生成
            print(f"已有匹配热图，年龄{age}")
        # 读取配准后的模板热图
        registered_template_label = nb.load(template_label_register_to_FeTA_file_path).get_fdata()
        # 反向配准后的模板热图大小放缩
        cropped_template_heatmap_data = crop_template_label(registered_template_label, data)


    # 应用于一张图片的11个通道并保存结果
    # 热图得到定位坐标和所在切片
    point_number_in_channel = 2
    orientations = [2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    location_image_names = ['1+2', '3+4', '5+6', '7+8', '9+10', '11+12', '13+14', '15+16', '17+18', '19+20', '21+22']
    template_heatmap_number = [1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 13, 14, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30]
    coordinates = []
    images_ups = []
    aspect_ratios = []
    cross_positions_list = []


    # 创建或打开一个文件用于写入
    with open(os.path.join(target_results_path, 'landmarks.txt'), 'w') as file:
        # 第一次遍历：计算并缓存所有需要的数据
        for i in range(11):
            # 约束，输入时给出通道对应的点
            heatmap_data_1 = constrain_heatmap(cropped_template_heatmap_data, heatmap[2*i], template_heatmap_number[2*i], radius[i], template_label_scale, template_label_pan)
            heatmap_data_2 = constrain_heatmap(cropped_template_heatmap_data, heatmap[2*i+1], template_heatmap_number[2*i+1], radius[i], template_label_scale, template_label_pan)
            # 保存约束后的热图 #####
            # nifti_image_1 = nb.Nifti1Image(heatmap_data_1, img_affine)
            # constrained_heatmap_file_path_1 = os.path.join(target_results_path, f'constrained_TTA_{2*i+1}.nii.gz')
            # nb.save(nifti_image_1, constrained_heatmap_file_path_1)
            # nifti_image_2 = nb.Nifti1Image(heatmap_data_2, img_affine)
            # constrained_heatmap_file_path_2 = os.path.join(target_results_path, f'constrained_TTA_{2*(i+1)}.nii.gz')
            # nb.save(nifti_image_2, constrained_heatmap_file_path_2)

            # 获取坐标和最大切片索引
            coordinate_1, slice_sum_max_index_1 = get_landmark_from_heatmap(heatmap_data_1, orientations[i], init_temp=init_temp, topk=topk)
            coordinate_2, slice_sum_max_index_2 = get_landmark_from_heatmap(heatmap_data_2, orientations[i], init_temp=init_temp, topk=topk)
            coordinate = np.array([coordinate_1, coordinate_2])
            slice_sum_max_index = round((slice_sum_max_index_1 + slice_sum_max_index_2) / 2)
            # print(coordinate)
            coordinates.append(coordinate)
            # 缓存切片图像
            img_ups, cross_positions = plot_slice(sized_mask, sized_data, slice_sum_max_index, orientations[i], max_value, min_value, coordinate)
            # 打印
            print(location_image_names[i], coordinate, slice_sum_max_index)

            # 每个通道有两个点，因此有六个坐标值
            merged_coordinates = coordinate.flatten()
            coordinates_str = ' '.join(map(str, merged_coordinates))
            # 写入文件
            file.write(f"{coordinates_str}\n")

            # 旋转图片
            img_ups = np.rot90(img_ups, k=1)
            # 旋转坐标
            rotated_cross_positions = []
            for (x, y) in cross_positions:
                x_new, y_new = rotate_coordinate(x, y, k=3, img_width=img_ups.shape[1], img_height=img_ups.shape[0])
                rotated_cross_positions.append((x_new, y_new))
            cross_positions = rotated_cross_positions

            # 左右削边
            non_zero_indices = np.nonzero(img_ups)
            leftmost = np.min(non_zero_indices[1])  # 最小列索引
            rightmost = np.max(non_zero_indices[1])  # 最大列索引
            new_width = rightmost - leftmost + 20  # 两侧各留出 10 像素
            h, w = img_ups.shape[:2]  # 获取原图像的高度和宽度
            # 创建新的空白图像（黑色背景）
            new_img = np.zeros((h, new_width, 3), dtype=img_ups.dtype)
            # 将原图像粘贴到新图像的中间部分
            paste_left = 10  # 左侧留出 10 像素
            new_img[:, paste_left:paste_left + (rightmost - leftmost)] = img_ups[:, leftmost:rightmost]
            # 更新图像数据
            img_ups = new_img
            # 更新坐标位置
            cross_positions = [(x, y-leftmost+paste_left) for (x, y) in cross_positions]
            # 缓存
            images_ups.append(img_ups)
            cross_positions_list.append(cross_positions)
            # 计算旋转后的宽高比（宽度/高度）
            h, w = img_ups.shape[:2]
            aspect_ratios.append(w / h)

    # 设置全局参数
    total_width_ratio = sum(aspect_ratios)  # 总宽度（相对单位）
    fig_height = 3  # 固定高度（英寸）
    wspace = -0.05  # 可调整的间距参数（相对单位）
    # 创建GridSpec布局
    fig = plt.figure(figsize=(fig_height * (total_width_ratio + wspace * 10), fig_height))  # 动态计算总宽度
    gs = gridspec.GridSpec(1, 11, width_ratios=aspect_ratios, wspace=wspace)

    # 第二次遍历：左右削边，使用缓存的数据绘制子图
    for i in range(11):  # 注意这里是从 0 到 10
        # 获取缓存的图像
        img_ups = images_ups[i]
        cross_positions = cross_positions_list[i]
        # 检查并调整图像数据
        if np.max(img_ups) > 1 or np.min(img_ups) < 0:  # 如果值范围不在 [0, 1]
            img_ups = (img_ups - np.min(img_ups)) / (np.max(img_ups) - np.min(img_ups))  # 归一化到 [0, 1]
        img_ups = img_ups.astype(np.float32)  # 确保数据类型正确
        # 创建子图
        ax = fig.add_subplot(gs[i])
        ax.imshow(img_ups, aspect='auto')
        # 在图像上绘制黑叉
        for (x, y) in cross_positions:
            ax.plot(y, x, 'kx', markersize=7, markeredgewidth=4)  # 绘制红色加号
        # 在图像上绘制红叉
        for (x, y) in cross_positions:
            ax.plot(y, x, 'rx', markersize=6, markeredgewidth=2)  # 绘制红色加号
        ax.axis('off')
        ax.set_adjustable('datalim')

    # 保存图像
    plt.savefig(target_results_path + '/landmarks.png', bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()