import cv2
import time

class ShotBoundaryDetector:
    def __init__(self, threshold=110000):
        self.threshold = threshold

#直方圖差異值
    def histogram_diff(self, hist1, hist2):
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    
#鏡頭變化的幀數列表
    def detect_shot_boundaries(self, video_path):
        cap = cv2.VideoCapture(video_path)
        prev_frame = None
        shot_boundaries = []
        frame_number = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]) # 計算灰度直方圖

            if prev_frame is not None:
                diff = self.histogram_diff(prev_hist, hist)
                if diff > self.threshold:   # 如果直方圖差異超過閾值，表示鏡頭變化
                    shot_boundaries.append(frame_number)

            prev_frame = frame # 更新前一幀
            prev_hist = hist # 更新前一幀的直方圖

        cap.release()
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        return shot_boundaries

    def format_shot_boundaries(self, shot_boundaries):
        formatted_boundaries = []
        start = None
        end = None
        
        for i in range(len(shot_boundaries)):
            if start is None:
                start = shot_boundaries[i]
                end = shot_boundaries[i]
            elif shot_boundaries[i] == end + 1:
                end = shot_boundaries[i]
            else:
                if start == end:
                    formatted_boundaries.append(start)
                else:
                    formatted_boundaries.append((start, end))
                start = shot_boundaries[i]
                end = shot_boundaries[i]
        
        if start is not None:
            if start == end:
                formatted_boundaries.append(start)
            else:
                formatted_boundaries.append((start, end))
        
        return formatted_boundaries
#打印鏡頭變化時間間隔
    def print_transitions(self, shot_boundaries):
        print("Shot transition intervals (frames):")
        formatted_boundaries = self.format_shot_boundaries(shot_boundaries)
        for boundary in formatted_boundaries:
            if isinstance(boundary, tuple):
                print(f"{boundary[0]}~{boundary[1]}")
            else:
                print(boundary)

# Example usage
video_path = 'climate.mp4'
detector = ShotBoundaryDetector()  # 創建鏡頭變化檢測器
shot_boundaries = detector.detect_shot_boundaries(video_path)# 檢測鏡頭變化
detector.print_transitions(shot_boundaries)# 打印鏡頭變化時間間隔
