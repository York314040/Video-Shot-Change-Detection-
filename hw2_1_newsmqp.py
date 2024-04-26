import cv2  
import numpy as np  
import time 

class ShotBoundaryDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.diff_threshold = 21  # 幀差異閾值
        self.gray_similarity_threshold = 0.1  # 灰度相似性閾值（未使用）
        self.color_similarity_threshold = 0.5  # 色彩相似性閾值（未使用）

    # 計算兩幀之間的絕對差異
    def calculate_frame_difference(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # 將幀轉換為灰度
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)  # 將幀轉換為灰度
        diff = cv2.absdiff(gray1, gray2)  # 計算幀之間的絕對差異
        norm_diff = np.sum(diff) / diff.size  # 歸一化絕對差異
        return norm_diff

    # 尋找鏡頭變化
    def find_shot_boundaries(self):
        start_time = time.time()  # 記錄開始時間
        cap = cv2.VideoCapture(self.video_path)  # 讀取視頻文件
        if not cap.isOpened():  # 檢查視頻文件是否成功打開
            print("無法打開視頻文件。")  # 打印錯誤消息
            return []  

        ret, prev_frame = cap.read()  # 讀取第一幀
        if not ret:  
            print("無法讀取視頻文件。")  
            return [] 

        shot_boundaries = []  # 初始化鏡頭變化列表
        frame_index = 1  # 幀索引
        start_frame = None  # 起始幀

        while True:
            ret, frame = cap.read()  # 讀取下一幀
            if not ret:  # 如果無法讀取下一幀，結束循環
                if start_frame is not None: 
                    shot_boundaries.append((start_frame, frame_index - 1))  # 將最後一個鏡頭變化加入列表
                break  

            frame_diff = self.calculate_frame_difference(prev_frame, frame)  # 計算幀之間的絕對差異

            if frame_diff > self.diff_threshold:  # 如果絕對差異超過閾值
                if start_frame is None:  # 如果是鏡頭變化的開始
                    start_frame = frame_index  
            else:  # 如果絕對差異未超過閾值
                if start_frame is not None:  # 如果已經開始檢測到鏡頭變化
                    shot_boundaries.append((start_frame, frame_index - 1))  # 將鏡頭變化範圍添加到列表
                    start_frame = None  # 重置起始幀

            prev_frame = frame  # 更新前一幀為當前幀
            frame_index += 1  # 增加幀索引

        cap.release()  # 釋放視頻捕獲資源
        end_time = time.time()  # 記錄結束時間
        print(f"Execution time:{end_time - start_time:.2f} seconds")  # 列印執行時間
        return shot_boundaries  # 返回鏡頭變化列表  

    # 列印鏡頭變化的時間間隔
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
