import cv2
import numpy as np
import time

class ShotBoundaryDetector:
    def __init__(self, threshold=110000):
        self.threshold = threshold

    # Calculate the difference between two histograms
    def histogram_diff(self, hist1, hist2):
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

    # Detect shot boundaries in a video
    def detect_shot_boundaries(self, video_path):
        start_time = time.time()
        cap = cv2.VideoCapture(video_path)
        prev_frame = None
        shot_boundaries = []
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

            if prev_frame is not None:
                diff = self.histogram_diff(prev_hist, hist)
                if diff > self.threshold:
                    shot_boundaries.append(frame_number)

            prev_frame = frame
            prev_hist = hist

        cap.release()
        end_time = time.time()
        print(f"Detection run time: {end_time - start_time:.2f} seconds")
        return shot_boundaries

# Read shot boundaries from a text file
def read_shot_boundaries_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        boundaries_from_txt = set()
        for line in lines:
            line = line.strip()
            if line.isdigit():
                boundaries_from_txt.add(int(line))
    return boundaries_from_txt

# Compare detected boundaries with boundaries from the text file
def compare_boundaries(detected_boundaries, txt_boundaries):
    detected_set = set(detected_boundaries)
    correct_matches = detected_set.intersection(txt_boundaries)
    print("\nComparing detected boundaries with TXT boundaries:")
    for boundary in detected_boundaries:
        match_status = "Match found" if boundary in correct_matches else "No match"
        print(f"{boundary}: {match_status}")
    print(f"\nCorrect matches: {len(correct_matches)}/{len(detected_boundaries)}")

# Main 
video_path = 'climate.mp4' 
txt_path = 'climate_ground.txt'  

start_time = time.time() 
detector = ShotBoundaryDetector()
shot_boundaries = detector.detect_shot_boundaries(video_path)
txt_boundaries = read_shot_boundaries_from_txt(txt_path)
compare_boundaries(shot_boundaries, txt_boundaries)
end_time = time.time() 