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
my_data_range = [(1, 1)]
data_range = [i for start, end in my_data_range for i in range(start, end + 1)]

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, 'data', 'cms50')
final_path = os.path.join(script_dir, 'data', 'dataset_Final')
os.makedirs(final_path, exist_ok=True)

# Mediapipe Face Detection & Face Mesh 초기화
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

def extract_face_cheek_roi(frame, size=(64, 64)):
    """ 얼굴을 먼저 검출한 후 뺨(cheek) ROI를 추출하고 입과 코를 제거한 후 64x64로 변환 """
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
stride = T // 3  # 슬라이딩 윈도우 간격

X_combined, Y_combined = [], []

for data_num in data_range:
    print(f"★ Processing data number ★ : {data_num}")
    
    time_file = os.path.join(base_path, f"{data_num}_time.txt")
    ppg_file = os.path.join(base_path, f"{data_num}_ppg.txt")
    nir_video_file = os.path.join(base_path, f"{data_num}_ir.mp4")
    
    if not all(os.path.exists(f) for f in [time_file, ppg_file, nir_video_file]):
        print(f"Missing file: {data_num}")
        continue
    
    time_data = np.loadtxt(time_file)
    ppg_data = pd.read_csv(ppg_file, sep=r'\s+', header=None)
    ppg_time, ppg_signals = ppg_data.iloc[1:, 0].values, ppg_data.iloc[1:, 1].values
    ppg_interp = interp1d(ppg_time, ppg_signals, kind='cubic', fill_value="extrapolate")
    
    cap_nir = cv2.VideoCapture(nir_video_file)
    frames_nir = []
    while True:
        ret_nir, frame_nir = cap_nir.read()
        if not ret_nir:
            break
        frames_nir.append(frame_nir)
    cap_nir.release()
    
    for start_idx in range(0, len(time_data) - T + 1, stride):
        clip_frames_nir = frames_nir[start_idx:start_idx + T]
        if len(clip_frames_nir) < T:
            continue
        
        roi_frames_combined = []
        ppg_values = []
        missing_faces = 0
        
        for idx in range(T):
            face_nir = extract_face_cheek_roi(clip_frames_nir[idx])
            ppg_value = ppg_interp(time_data[start_idx + idx])
            
            if face_nir is None:
                missing_faces += 1
                continue
            if missing_faces > T * 0.1:
                break
            
            face_nir_gray = cv2.cvtColor(face_nir, cv2.COLOR_BGR2GRAY)
            face_nir_gray = face_nir_gray[..., np.newaxis]
            
            roi_frames_combined.append(face_nir_gray)
            ppg_values.append(ppg_value)
        
        if len(roi_frames_combined) == T:
            X_combined.append(np.array(roi_frames_combined))
            Y_combined.append(np.array(ppg_values))

X_combined = np.array(X_combined)
Y_combined = np.array(Y_combined)

np.save(os.path.join(final_path, "X_dataset.npy"), X_combined)
np.save(os.path.join(final_path, "Y_dataset.npy"), Y_combined)

print("★★★ 데이터 저장 완료! ★★★")
print("X_data shape:", X_combined.shape)
print("Y_data shape:", Y_combined.shape)
