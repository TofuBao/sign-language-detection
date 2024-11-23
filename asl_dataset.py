import torch
import torch.nn as nn
import numpy as np
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def ensure_sequence_shape(sequence):
    """
    确保输入的 sequence 形状为 (seq_len, n_keypoints, 3)。
    如果输入是 2D，将其重塑为 3D。
    """
    if sequence.dim() == 2 and sequence.shape[1] % 3 == 0:
        n_keypoints = sequence.shape[1] // 3
        sequence = sequence.view(sequence.shape[0], n_keypoints, 3)
    elif sequence.dim() != 3:
        raise ValueError(f"Expected sequence to have shape (seq_len, n_keypoints, 3), but got {sequence.shape}")
    return sequence
def compute_mean_std(X_train):
    """
    计算数据集 X_train 的全局均值和标准差。
    
    Args:
        X_train (list or np.ndarray): 包含所有训练样本的列表或数组。
    
    Returns:
        tuple: global_mean, global_std
    """
    all_keypoints = np.concatenate([sample for sample in X_train], axis=0)  # 将所有样本拼接在一起
    mean = np.nanmean(all_keypoints, axis=0)  # 忽略 NaN 计算均值
    std = np.nanstd(all_keypoints, axis=0)    # 忽略 NaN 计算标准差
    return mean, std
def temporal_resample(sequence, target_length=105, min_scale=0.5, max_scale=1.5):
    sequence = ensure_sequence_shape(sequence)  # 确保序列形状为 (seq_len, n_keypoints, 3)
    scale = np.random.uniform(min_scale, max_scale)
    new_length = int(sequence.shape[0] * scale)

    # 调整为 3D 输入
    n_keypoints = sequence.shape[1]
    sequence = sequence.permute(1, 0, 2).contiguous().view(-1, sequence.shape[0])  # (n_keypoints * 3, seq_len)
    
    # 重采样到 new_length
    resampled_sequence = F.interpolate(sequence.unsqueeze(0), size=new_length, mode='linear', align_corners=False).squeeze(0)
    
    # 确保重塑大小匹配，并恢复 (new_length, n_keypoints, 3) 的形状
    resampled_sequence = resampled_sequence.view(n_keypoints, new_length, 3).permute(1, 0, 2)

    # 最终调整到 target_length
    final_sequence = F.interpolate(resampled_sequence.permute(1, 2, 0).contiguous().view(-1, resampled_sequence.shape[0]).unsqueeze(0),
                                   size=target_length, mode='linear', align_corners=False).squeeze(0)
    
    # 恢复形状 (target_length, n_keypoints, 3)
    final_sequence = final_sequence.view(n_keypoints, target_length, 3).permute(1, 0, 2)
    return final_sequence

def windowed_cutmix(sequence1, sequence2, target_length=105):
    sequence1, sequence2 = ensure_sequence_shape(sequence1), ensure_sequence_shape(sequence2)
    cut_ratio = np.random.rand()
    cut_point1 = int(sequence1.shape[0] * cut_ratio)
    cut_point2 = int(sequence2.shape[0] * cut_ratio)
    mixed_sequence = torch.cat((sequence1[:cut_point1], sequence2[cut_point2:]), dim=0)

    # 调整为 3D 输入，应用 interpolate
    mixed_sequence = mixed_sequence.permute(1, 0, 2).contiguous()  # (n_keypoints, seq_len, 3)
    mixed_sequence = mixed_sequence.view(-1, mixed_sequence.shape[1])  # (n_keypoints * 3, seq_len)
    final_sequence = F.interpolate(mixed_sequence.unsqueeze(0), size=target_length, mode='linear', align_corners=False).squeeze(0)
    final_sequence = final_sequence.view(sequence1.shape[1], target_length, 3).permute(1, 0, 2)  # (target_length, n_keypoints, 3)
    return final_sequence
def temporal_shift(sequence, target_length=105, max_shift=10):
    sequence = ensure_sequence_shape(sequence)  # 确保形状为 (seq_len, n_keypoints, 3)
    shift = np.random.randint(-max_shift, max_shift)
    shifted_sequence = torch.roll(sequence, shifts=shift, dims=0)

    # 最终调整为 target_length
    seq_len = shifted_sequence.shape[0]
    if seq_len > target_length:
        shifted_sequence = F.interpolate(shifted_sequence.permute(1, 2, 0).unsqueeze(0), size=target_length, mode='linear', align_corners=False)
        shifted_sequence = shifted_sequence.squeeze(0).permute(2, 0, 1)
    elif seq_len < target_length:
        padding = torch.zeros((target_length - seq_len, shifted_sequence.shape[1], shifted_sequence.shape[2]), dtype=shifted_sequence.dtype)
        shifted_sequence = torch.cat((shifted_sequence, padding), dim=0)
    
    return shifted_sequence

def random_keypoint_dropout(sequence, num_points_to_drop=6, num_time_windows=3):
    sequence = ensure_sequence_shape(sequence)  # 确保形状为 (seq_len, n_keypoints, 3)
    seq_len, num_keypoints, _ = sequence.shape
    for _ in range(num_time_windows):
        start = np.random.randint(0, seq_len)
        end = min(seq_len, start + np.random.randint(1, seq_len // num_time_windows))
        drop_indices = np.random.choice(num_keypoints, num_points_to_drop, replace=False)
        sequence[start:end, drop_indices, :] = 0  # 将选定关键点置为 0
    return sequence

def spatial_mask(sequence, mask_prob=0.3, max_points=10):
    sequence = ensure_sequence_shape(sequence)  # 确保形状为 (seq_len, n_keypoints, 3)
    if np.random.rand() < mask_prob:
        num_keypoints = sequence.shape[1]
        mask_points = np.random.choice(num_keypoints, max_points, replace=False)
        sequence[:, mask_points, :] = 0  # 空间遮挡，将选定的关键点置为 0
    return sequence

def temporal_mask(sequence, mask_prob=0.3, max_mask_len=10):
    sequence = ensure_sequence_shape(sequence)  # 确保形状为 (seq_len, n_keypoints, 3)
    if np.random.rand() < mask_prob:
        seq_len = sequence.shape[0]
        mask_len = np.random.randint(1, max_mask_len)
        start = np.random.randint(0, seq_len - mask_len)
        sequence[start:start + mask_len, :, :] = 0  # 时间遮挡，将选定时间窗口内的关键点置为 0
    return sequence

def drop_face_or_pose(sequence, drop_face_prob=0.2, drop_pose_prob=0.2, face_indices=None, pose_indices=None):
    sequence = ensure_sequence_shape(sequence)  # 确保形状为 (seq_len, n_keypoints, 3)
    if np.random.rand() < drop_face_prob and face_indices is not None:
        sequence[:, face_indices, :] = 0  # 面部关键点置为 0
    if np.random.rand() < drop_pose_prob and pose_indices is not None:
        sequence[:, pose_indices, :] = 0  # 姿态关键点置为 0
    return sequence

def drop_hand_keypoints(sequence, drop_hand_prob=0.05, left_hand_indices=None, right_hand_indices=None):
    sequence = ensure_sequence_shape(sequence)  # 确保形状为 (seq_len, n_keypoints, 3)
    if np.random.rand() < drop_hand_prob:
        if left_hand_indices is not None:
            sequence[:, left_hand_indices, :] = 0  # 左手关键点置为 0
        if right_hand_indices is not None:
            sequence[:, right_hand_indices, :] = 0  # 右手关键点置为 0
    return sequence

def flip_keypoints(sequence, left_hand_indices, right_hand_indices):
    """
    仅对左右手的关键点进行翻转。

    Args:
        sequence (torch.Tensor): 形状为 (seq_len, n_keypoints, 3) 的关键点序列。
        left_hand_indices (list): 左手关键点索引。
        right_hand_indices (list): 右手关键点索引。
    
    Returns:
        torch.Tensor: 翻转后的关键点序列。
    """
    sequence = ensure_sequence_shape(sequence)  # 确保形状为 (seq_len, n_keypoints, 3)

    # 翻转 X 坐标
    sequence[:, :, 0] = -sequence[:, :, 0]

    # 创建副本，用于交换左右手关键点
    flipped_sequence = sequence.clone()
    for l_idx, r_idx in zip(left_hand_indices, right_hand_indices):
        flipped_sequence[:, r_idx, :] = sequence[:, l_idx, :]  # 左手 → 右手
        flipped_sequence[:, l_idx, :] = sequence[:, r_idx, :]  # 右手 → 左手

    return flipped_sequence
def compute_velocity_and_acceleration(sequence):
    """
    计算给定序列的速度和加速度。
    Args:
        sequence (torch.Tensor): 形状为 (seq_len, n_keypoints, 3) 的位置序列。
    Returns:
        torch.Tensor, torch.Tensor: 分别返回速度和加速度，形状均为 (seq_len, n_keypoints, 3)。
    """
    sequence = ensure_sequence_shape(sequence)  # 确保输入是 (seq_len, n_keypoints, 3)
    
    # 计算速度 (Δx, Δy, Δz)
    velocity = torch.diff(sequence, dim=0, prepend=sequence[:1])
    
    # 计算加速度 (Δv_x, Δv_y, Δv_z)
    acceleration = torch.diff(velocity, dim=0, prepend=velocity[:1])
    
    return velocity, acceleration


def affine_transform(sequence, scale_range=(0.9, 1.1), translation_range=(-0.1, 0.1), rotation_range=(-10, 10)):
    sequence = ensure_sequence_shape(sequence)  # 确保形状为 (seq_len, n_keypoints, 3)

    # 随机缩放
    scale = np.random.uniform(*scale_range)
    sequence = sequence * scale

    # 随机平移
    translation = np.random.uniform(*translation_range, size=(1, sequence.shape[1], sequence.shape[2]))
    sequence = sequence + torch.tensor(translation, dtype=sequence.dtype, device=sequence.device)

    # 随机旋转（围绕 Z 轴旋转）
    angle = np.radians(np.random.uniform(*rotation_range))
    rotation_matrix = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ], dtype=sequence.dtype, device=sequence.device)
    sequence = torch.matmul(sequence, rotation_matrix)
    return sequence

# 在3D形状的 (seq_len, n_keypoints, 3) 中，定义关键点索引范围
face_indices = list(range(0, 90))            # 面部关键点在 3D 中为 0~89
pose_indices = list(range(90, 90 + 33))      # 姿态关键点在 3D 中为 90~122
left_hand_indices = list(range(123, 123 + 21))  # 左手关键点在 3D 中为 123~143
right_hand_indices = list(range(144, 144 + 21)) # 右手关键点在 3D 中为 144~164

DATA_PATH = "I:\\Ece496\\custom_data\\preprocess"
target_length = 105  # 设置目标长度

# 加载数据
X = np.load(os.path.join(DATA_PATH, "X.npy"), allow_pickle=True)
y = np.load(os.path.join(DATA_PATH, "y.npy"), allow_pickle=True)

# 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 保存划分后的数据
np.save(os.path.join(DATA_PATH, "X_train.npy"), X_train)
np.save(os.path.join(DATA_PATH, "y_train.npy"), y_train)
np.save(os.path.join(DATA_PATH, "X_val.npy"), X_val)
np.save(os.path.join(DATA_PATH, "y_val.npy"), y_val)
np.save(os.path.join(DATA_PATH, "X_test.npy"), X_test)
np.save(os.path.join(DATA_PATH, "y_test.npy"), y_test)

# 定义自定义数据集类
class ASLDataset(Dataset):
    def __init__(self, X, y, global_mean, global_std, augment=True):
        self.X = X
        self.y = y
        self.global_mean = global_mean
        self.global_std = global_std
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sequence = self.X[idx]
        label = self.y[idx]

        sequence = np.nan_to_num(sequence, nan=0.0)
        
        # 归一化和中心化
        sequence = (sequence - self.global_mean) / self.global_std
        sequence = torch.from_numpy(sequence).float()

        # 确保形状为 (seq_len, n_keypoints, 3)
        sequence = ensure_sequence_shape(sequence)

        # 应用数据增强
        if self.augment:
            sequence = apply_random_augmentations(sequence, target_length=target_length)

        return sequence, label

def apply_random_augmentations(sequence, augment_prob=0.5, resize_prob=0.8, target_length=105):
    """
    对序列数据应用增强操作，支持未展平的 3D 数据。
    sequence 形状: (seq_len, n_keypoints, feature_dim)
    """
    sequence = ensure_sequence_shape(sequence)  # 确保序列为 3D 形状 (seq_len, n_keypoints, 3)
    print(f"Initial sequence shape: {sequence.shape}")

    # 1. 初始计算速度和加速度（不添加到序列中）
    initial_velocity, initial_acceleration = compute_velocity_and_acceleration(sequence)

    # 2. 时间增强
    if np.random.rand() < augment_prob:
        sequence = temporal_resample(sequence, target_length=target_length)
        print(f"Sequence shape after temporal_resample: {sequence.shape}")
        # 重新计算速度和加速度
        velocity, acceleration = compute_velocity_and_acceleration(sequence)

    if np.random.rand() < augment_prob:
        sequence = temporal_shift(sequence)
        print(f"Sequence shape after temporal_shift: {sequence.shape}")
        # 重新计算速度和加速度
        velocity, acceleration = compute_velocity_and_acceleration(sequence)

    # 3. 空间增强
    if np.random.rand() < augment_prob:
        sequence = affine_transform(sequence)
        print(f"Sequence shape after affine_transform: {sequence.shape}")
        # 重新计算速度和加速度
        velocity, acceleration = compute_velocity_and_acceleration(sequence)

    # 4. 遮罩增强（无需重新计算速度和加速度）
    if np.random.rand() < augment_prob:
        sequence = drop_face_or_pose(sequence, face_indices=face_indices, pose_indices=pose_indices)
        print(f"Sequence shape after drop_face_or_pose: {sequence.shape}")
    if np.random.rand() < augment_prob:
        sequence = drop_hand_keypoints(sequence, left_hand_indices=left_hand_indices, right_hand_indices=right_hand_indices)
        print(f"Sequence shape after drop_hand_keypoints: {sequence.shape}")
    if np.random.rand() < augment_prob:
        sequence = spatial_mask(sequence)
        print(f"Sequence shape after spatial_mask: {sequence.shape}")
    if np.random.rand() < augment_prob:
        sequence = temporal_mask(sequence)
        print(f"Sequence shape after temporal_mask: {sequence.shape}")

    # 5. 翻转（需要重新计算速度和加速度）
    if np.random.rand() < augment_prob:
        sequence = flip_keypoints(sequence, left_hand_indices, right_hand_indices)
        print(f"Sequence shape after flip_keypoints: {sequence.shape}")
        # 重新计算速度和加速度
        velocity, acceleration = compute_velocity_and_acceleration(sequence)

    # 6. 最后将完整的速度和加速度加入序列
    if 'velocity' in locals() and 'acceleration' in locals():
        sequence = torch.cat((sequence, velocity, acceleration), dim=2)
    else:
        # 如果没有动态重新计算，使用初始速度和加速度
        sequence = torch.cat((sequence, initial_velocity, initial_acceleration), dim=2)
    
    print(f"Final sequence shape: {sequence.shape}")
    
    return sequence





# 计算均值和标准差用于归一化
global_mean, global_std = compute_mean_std(X_train)



# 创建数据集和数据加载器
train_dataset = ASLDataset(X_train, y_train, global_mean, global_std, augment=True)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# # 训练循环中应用增强并打印处理信息
# for epoch in range(1):  # 设定单个 epoch 测试
#     print(f"Epoch {epoch + 1}")
#     for batch_idx, (sequences, labels) in enumerate(train_loader):
#         print(f"Batch {batch_idx + 1}:")
#         for sample_idx, sequence in enumerate(sequences):
#             print(f"  Sample {sample_idx + 1} after augmentation has shape: {sequence.shape}")
#         break  # 只打印一个批次，确保代码运行后不输出过多信息
#     break
