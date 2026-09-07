import numpy as np

def normalize_landmarks(landmarks):
    """
    Normalizes a sequence of raw landmarks.
    landmarks shape: (T, 225) where:
      - 0:99 is Pose (33 points)
      - 99:162 is Left Hand (21 points)
      - 162:225 is Right Hand (21 points)
    
    Normalization steps:
    1. Center coordinates around the shoulder center (index 11 and 12 of Pose).
    2. Scale coordinates by shoulder distance.
    3. Center hand landmarks relative to wrists.
    """
    T, D = landmarks.shape
    normalized = landmarks.copy()
    
    for t in range(T):
        # Extract pose coordinates
        pose = normalized[t, 0:99].reshape(33, 3)
        
        # Shoulder indices: 11 (left), 12 (right)
        l_shoulder = pose[11]
        r_shoulder = pose[12]
        
        # Calculate shoulder center and width
        if np.all(l_shoulder == 0) or np.all(r_shoulder == 0):
            shoulder_center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
            shoulder_width = 1.0
        else:
            shoulder_center = (l_shoulder + r_shoulder) / 2.0
            shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
            if shoulder_width == 0:
                shoulder_width = 1.0
                
        # Center and scale pose
        pose = (pose - shoulder_center) / shoulder_width
        normalized[t, 0:99] = pose.flatten()
        
        # Extract hand coordinates
        lh = normalized[t, 99:162].reshape(21, 3)
        rh = normalized[t, 162:225].reshape(21, 3)
        
        # Center hands relative to wrists if hands are detected (i.e. non-zero)
        # Wrists are index 0 of hand landmarks
        if not np.all(lh == 0):
            lh_wrist = lh[0].copy()
            lh = (lh - lh_wrist) / shoulder_width
            
        if not np.all(rh == 0):
            rh_wrist = rh[0].copy()
            rh = (rh - rh_wrist) / shoulder_width
            
        normalized[t, 99:162] = lh.flatten()
        normalized[t, 162:225] = rh.flatten()
        
    return normalized

def extract_temporal_features(landmarks):
    """
    Computes first-order (velocity) and second-order (acceleration) temporal differences.
    landmarks shape: (T, D)
    Returns:
      velocities: (T, D)
      accelerations: (T, D)
    """
    T, D = landmarks.shape
    
    # Velocities: v(t) = x(t) - x(t-1)
    velocities = np.zeros_like(landmarks)
    if T > 1:
        velocities[1:] = landmarks[1:] - landmarks[:-1]
        velocities[0] = velocities[1] # boundary pad
        
    # Accelerations: a(t) = v(t) - v(t-1)
    accelerations = np.zeros_like(velocities)
    if T > 1:
        accelerations[1:] = velocities[1:] - velocities[:-1]
        accelerations[0] = accelerations[1] # boundary pad
        
    return velocities, accelerations
