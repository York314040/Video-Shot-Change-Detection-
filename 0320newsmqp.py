import cv2
import numpy as np
import time

class ShotBoundaryDetector:
    
    def __init__(self, video_path):
        self.video_path = video_path  
        self.diff_threshold = 21  

    # Calculate the difference between the two frames
    def calculate_frame_difference(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        norm_diff = np.sum(diff) / diff.size
        return norm_diff

    #  Detect lens boundaries in video
    def find_shot_boundaries(self):
        start_time = time.time()  
        cap = cv2.VideoCapture(self.video_path)  
        if not cap.isOpened(): 
            print("無法打開影片")
            return []

        ret, prev_frame = cap.read() 
        if not ret:  
            print("無法讀取影片")
            return []

        shot_boundaries = []  
        frame_index = 1 

        while True:
            ret, frame = cap.read()  
            if not ret:  
                break

            frame_diff = self.calculate_frame_difference(prev_frame, frame)  

            if frame_diff > self.diff_threshold:  # If the difference exceeds the threshold, it is recorded as a shot boundary
                shot_boundaries.append(frame_index)

            prev_frame = frame 
            frame_index += 1 

        cap.release()  
        end_time = time.time()  
        print(f"\nExecution time: {end_time - start_time:.2f} seconds")  
        return shot_boundaries  

    # Prints the shot change frame number
    def print_transitions(self, shot_boundaries):
        print("鏡頭變換幀數：")
        for boundary in shot_boundaries:
            print(boundary)

# Read shot boundaries from a text file
def read_shot_boundaries_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        boundaries_from_txt = []
        for line in lines:
            line = line.strip()  
    
            if not line.replace('~', '').replace(' ', '').isdigit():
                continue
           
            if '~' in line:
                start, end = map(int, line.split('~'))
                boundaries_from_txt.extend(list(range(start, end + 1)))
            else:  
                boundaries_from_txt.append(int(line))
    return boundaries_from_txt  

# Compare the detected lens boundaries with those in the text file
def compare_boundaries(detected_boundaries, txt_boundaries):
    print("\nComparing detected boundaries with TXT boundaries:")
    correct = 0
    for boundary in detected_boundaries:
        if boundary in txt_boundaries:
            print(f"Match found: {boundary}")  
            correct += 1
        else:
            print(f"No match for: {boundary}")  
    print(f"\nCorrect matches: {correct}/{len(detected_boundaries)}") 

# main
video_path = 'news.mpg'  
txt_path = 'news_ground.txt'  

detector = ShotBoundaryDetector(video_path)  
shot_boundaries = detector.find_shot_boundaries()  
txt_boundaries = read_shot_boundaries_from_txt(txt_path) 
compare_boundaries(shot_boundaries, txt_boundaries) 
