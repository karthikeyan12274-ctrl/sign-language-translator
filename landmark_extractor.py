import cv2
import numpy as np
import mediapipe as mp

class LandmarkExtractor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract_frame_landmarks(self, frame):
        """
        Extracts pose, left hand, and right hand landmarks from a single RGB frame.
        Returns a flat numpy array of shape (225,)
        """
        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        
        # 1. Pose landmarks (33 points * 3 = 99 values)
        pose_coords = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                pose_coords.extend([landmark.x, landmark.y, landmark.z])
        else:
            pose_coords = [0.0] * 99

        # 2. Left Hand landmarks (21 points * 3 = 63 values)
        lh_coords = []
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                lh_coords.extend([landmark.x, landmark.y, landmark.z])
        else:
            lh_coords = [0.0] * 63

        # 3. Right Hand landmarks (21 points * 3 = 63 values)
        rh_coords = []
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                rh_coords.extend([landmark.x, landmark.y, landmark.z])
        else:
            rh_coords = [0.0] * 63

        # Concatenate all coordinates: shape (225,)
        all_landmarks = np.array(pose_coords + lh_coords + rh_coords, dtype=np.float32)
        return all_landmarks

    def extract_video_landmarks(self, video_path):
        """
        Extract landmarks from a whole video file.
        Returns a numpy array of shape (num_frames, 225)
        """
        cap = cv2.VideoCapture(video_path)
        sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = self.extract_frame_landmarks(frame)
            sequence.append(landmarks)
            
        cap.release()
        return np.array(sequence, dtype=np.float32)
        
    def close(self):
        self.holistic.close()
