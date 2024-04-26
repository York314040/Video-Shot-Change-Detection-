import cv2
import numpy as np
import time

class ShotBoundaryDetector:
    def __init__(self, video_path, threshold=110000, diff_threshold=21):
        self.video_path = video_path
        self.threshold = threshold
        self.diff_threshold = diff_threshold

    def calculate_frame_difference(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        norm_diff = np.sum(diff) / diff.size
        return norm_diff

    def histogram_diff(self, hist1, hist2):
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

    def detect_shot_boundaries(self):
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
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
        start_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                if start_frame is not None:
                    shot_boundaries.append((start_frame, frame_index - 1))
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

            frame_diff = self.calculate_frame_difference(prev_frame, frame)
            hist_diff = self.histogram_diff(prev_hist, hist)

            if frame_diff > self.diff_threshold or hist_diff > self.threshold:
                if start_frame is None:
                    start_frame = frame_index
            else:
                if start_frame is not None:
                    shot_boundaries.append((start_frame, frame_index - 1))
                    start_frame = None

            prev_frame = frame
            prev_hist = hist
            frame_index += 1

        cap.release()
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        return shot_boundaries

    def format_shot_boundaries(self, shot_boundaries):
        formatted_boundaries = []
        start = None
        end = None

        for boundary in shot_boundaries:
            if start is None:
                start = boundary[0]
                end = boundary[1]
            elif boundary[0] == end + 1:
                end = boundary[1]
            else:
                if start == end:
                    formatted_boundaries.append(start)
                else:
                    formatted_boundaries.append((start, end))
                start = boundary[0]
                end = boundary[1]

        if start is not None:
            if start == end:
                formatted_boundaries.append(start)
            else:
                formatted_boundaries.append((start, end))

        return formatted_boundaries

    def print_transitions(self, shot_boundaries):
        print("Shot transition intervals (frames):")
        formatted_boundaries = self.format_shot_boundaries(shot_boundaries)
        for boundary in formatted_boundaries:
            if isinstance(boundary, tuple):
                print(f"{boundary[0]}~{boundary[1]}")
            else:
                print(boundary)

# Example usage
video_path = 'ngc.mpeg'
detector = ShotBoundaryDetector(video_path)
shot_boundaries = detector.detect_shot_boundaries()
detector.print_transitions(shot_boundaries)
