import os
import numpy as np
import matplotlib.pyplot as plt

# 데이터셋 로드
script_dir = os.path.dirname(os.path.abspath(__file__))
final_path = os.path.join(script_dir, 'data', 'dataset_Final')

X_data = np.load(os.path.join(final_path, "X_dataset.npy"))

# 랜덤 샘플 및 프레임 선택
sample_idx = np.random.randint(0, X_data.shape[0])  # 랜덤한 샘플 선택
frame_idx = np.random.randint(0, X_data.shape[1])  # 해당 샘플에서 랜덤한 프레임 선택

sample_frame = X_data[sample_idx, frame_idx]  # (64, 64, 4) 형태의 데이터

# RGB와 NIR 분리
rgb_image = sample_frame[:, :, :3]  # RGB 채널 (64, 64, 3)
nir_image = sample_frame[:, :, 3]   # NIR 채널 (64, 64)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(rgb_image.astype(np.uint8))
axes[0].set_title("RGB Image")
axes[0].axis("off")

axes[1].imshow(nir_image, cmap="gray")
axes[1].set_title("NIR Image")
axes[1].axis("off")

plt.show()
