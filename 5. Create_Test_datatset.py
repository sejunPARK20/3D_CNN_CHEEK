import os
import cv2
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
import mediapipe as mp

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 데이터 범위 설정
my_data_range = [(0, 0)]
data_range = [i for start, end in my_data_range for i in range(start, end + 1)]

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, 'data', 'test_data')
final_path = os.path.join(script_dir, 'data', 'dataset_Test_Final')
os.makedirs(final_path, exist_ok=True)

# Mediapipe 초기화
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# 뺨(cheek) Landmarks
cheek_landmarks = [
    152, 377, 400, 378, 365, 367, 435, 401, 366, 447, 346, 347, 329, 
    118, 117, 227, 137, 177, 215, 138, 136, 149, 176, 148
]

# 입과 코 부분 Landmark (제거할 영역)
mouth_nose_landmarks = [
    17, 314, 273, 287, 279, 456, 6, 236, 49, 57, 43, 84
]

def extract_cheek_roi(frame, size=(64, 64)):
    """ 얼굴에서 뺨(cheek) 부분만 추출하여 64x64로 변환 """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_detection.process(frame_rgb)
    h, w, _ = frame.shape
    
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            face_roi = frame[y:y+h_box, x:x+w_box]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            mesh_results = face_mesh.process(face_rgb)
            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    mask = np.zeros_like(face_roi[:, :, 0])
                    points_cheek = np.array([
                        (int(face_landmarks.landmark[idx].x * w_box),
                         int(face_landmarks.landmark[idx].y * h_box)) for idx in cheek_landmarks
                    ], np.int32)
                    points_mouth_nose = np.array([
                        (int(face_landmarks.landmark[idx].x * w_box),
                         int(face_landmarks.landmark[idx].y * h_box)) for idx in mouth_nose_landmarks
                    ], np.int32)
                    
                    if len(points_cheek) > 2:
                        cv2.fillPoly(mask, [cv2.convexHull(points_cheek)], 255)
                    if len(points_mouth_nose) > 2:
                        cv2.fillPoly(mask, [cv2.convexHull(points_mouth_nose)], 0)
                    
                    cheek_roi = cv2.bitwise_and(face_roi, face_roi, mask=mask)
                    cheek_roi_resized = cv2.resize(cheek_roi, size)
                    return cheek_roi_resized
    return None

T = 300  # 한 클립당 프레임 수

X_combined, Y_combined = [], []

for data_num in data_range:
    print(f"★ Processing data number ★ : {data_num}")
    
    time_file = os.path.join(base_path, f"{data_num}_time.txt")
    ppg_file = os.path.join(base_path, f"{data_num}_ppg.txt")
    nir_video_file = os.path.join(base_path, f"{data_num}_ir.mp4")
    rgb_video_file = os.path.join(base_path, f"{data_num}_rgb.mp4")
    
    if not all(os.path.exists(f) for f in [time_file, ppg_file, nir_video_file, rgb_video_file]):
        print(f"Missing file: {data_num}")
        continue
    
    time_data = np.loadtxt(time_file)
    ppg_data = pd.read_csv(ppg_file, sep=r'\s+', header=None)
    ppg_time, ppg_signals = ppg_data.iloc[1:, 0].values, ppg_data.iloc[1:, 1].values
    ppg_interp = interp1d(ppg_time, ppg_signals, kind='cubic', fill_value="extrapolate")
    
    cap_nir = cv2.VideoCapture(nir_video_file)
    cap_rgb = cv2.VideoCapture(rgb_video_file)
    frames_nir, frames_rgb = [], []
    while True:
        ret_nir, frame_nir = cap_nir.read()
        ret_rgb, frame_rgb = cap_rgb.read()
        if not ret_nir or not ret_rgb:
            break
        frames_nir.append(frame_nir)
        frames_rgb.append(frame_rgb)
    cap_nir.release()
    cap_rgb.release()
    
    for start_idx in [0, 300, 600, 900]:
        clip_frames_rgb = frames_rgb[start_idx:start_idx + T]
        clip_frames_nir = frames_nir[start_idx:start_idx + T]
        if len(clip_frames_rgb) < T or len(clip_frames_nir) < T:
            continue
        
        roi_frames_combined = []
        ppg_values = []
        for idx in range(T):
            face_rgb = extract_cheek_roi(clip_frames_rgb[idx])
            face_nir = extract_cheek_roi(clip_frames_nir[idx])
            if face_rgb is None or face_nir is None:
                continue
            
            face_nir_gray = cv2.cvtColor(face_nir, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
            combined_frame = np.concatenate([face_rgb, face_nir_gray], axis=-1)[:,:,:4]
            roi_frames_combined.append(combined_frame)
            ppg_values.append(ppg_interp(time_data[start_idx + idx]))
        
        if len(roi_frames_combined) == T:
            X_combined.append(np.array(roi_frames_combined))
            Y_combined.append(np.array(ppg_values))

np.save(os.path.join(final_path, "X_dataset_test.npy"), np.array(X_combined))
np.save(os.path.join(final_path, "Y_dataset_test.npy"), np.array(Y_combined))

print("★★★ 데이터 저장 완료! ★★★")
print("X_data shape:", np.array(X_combined).shape)
print("Y_data shape:", np.array(Y_combined).shape)