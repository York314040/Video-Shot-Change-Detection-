import cv2  
import numpy as np  
import time 

class ShotBoundaryDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.diff_threshold = 21  # 幀差異閾值
        self.gray_similarity_threshold = 0.1
        self.color_similarity_threshold = 0.5
        self.flow_threshold = 0.5  # 光流閾值，根據需求調整

    def calculate_frame_difference(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        norm_diff = np.sum(diff) / diff.size
        return norm_diff

    def calculate_optical_flow(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # 計算光流的幅度和角度
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # 計算幅度的平均值作為光流變化的量化指標
        mean_magnitude = np.mean(magnitude)
        return mean_magnitude

    def find_shot_boundaries(self):
        start_time = time.time()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("無法打開視頻文件。")
            return []

        ret, prev_frame = cap.read()
        if not ret:
            print("無法讀取視頻文件。")
            return []

        shot_boundaries = []
        frame_index = 1
        start_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                if start_frame is not None:
                    shot_boundaries.append((start_frame, frame_index - 1))
                break

            frame_diff = self.calculate_frame_difference(prev_frame, frame)
            flow_diff = self.calculate_optical_flow(prev_frame, frame)

            if frame_diff > self.diff_threshold or flow_diff > self.flow_threshold:
                if start_frame is None:
                    start_frame = frame_index
            else:
                if start_frame is not None:
                    shot_boundaries.append((start_frame, frame_index - 1))
                    start_frame = None

            prev_frame = frame
            frame_index += 1

        cap.release()
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        return shot_boundaries

    def print_transitions(self, shot_boundaries):
        print("Shot transition intervals (frames):")
        for start, end in shot_boundaries:
            if start == end:
                print(f"{start}")
            else:
                print(f"{start}~{end}")


video_path = 'ngc.mpeg'
detector = ShotBoundaryDetector(video_path)
shot_boundaries = detector.find_shot_boundaries()
detector.print_transitions(shot_boundaries)
