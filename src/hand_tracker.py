import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        # Drawing styles
        self.hand_connection_style = self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.hand_landmark_style = self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=3)

    def process_frame(self, frame_bgr):
        """
        Processes a BGR OpenCV frame.
        Returns:
            processed_frame: frame with landmarks drawn
            cursor_pos: (x, y) coordinates of index finger (scaled to frame width/height) or None
            gesture: string ('draw', 'hover', 'clear', 'none')
        """
        # 1. Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Process image with MediaPipe Hands
        results = self.hands.process(frame_rgb)
        
        cursor_pos = None
        gesture = 'none'
        
        h, w, c = frame_bgr.shape
        processed_frame = frame_bgr.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the output frame
                self.mp_draw.draw_landmarks(
                    processed_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.hand_landmark_style,
                    self.hand_connection_style
                )
                
                # Extract landmarks coordinates
                # Landmarks are normalized (0.0 to 1.0) relative to image width/height
                landmarks = hand_landmarks.landmark
                
                # Check finger states (up vs down)
                # True if finger is extended (tip is higher than pip joint in Y axis, remember Y decreases upwards)
                # joint indices: 
                # Thumb: tip 4, ip 3, mcp 2
                # Index: tip 8, pip 6
                # Middle: tip 12, pip 10
                # Ring: tip 16, pip 14
                # Pinky: tip 20, pip 18
                
                index_up = landmarks[8].y < landmarks[6].y and landmarks[8].y < landmarks[5].y
                middle_up = landmarks[12].y < landmarks[10].y and landmarks[12].y < landmarks[9].y
                ring_up = landmarks[16].y < landmarks[14].y and landmarks[16].y < landmarks[13].y
                pinky_up = landmarks[20].y < landmarks[18].y and landmarks[20].y < landmarks[17].y
                
                # Thumb is a bit different (horizontal movement mostly, check x position relative to joint)
                # For simplicity, we can use y-axis relative to joint for classification
                thumb_up = landmarks[4].y < landmarks[3].y
                
                # Get coordinates of Index finger tip (landmark 8)
                index_tip_x = int(landmarks[8].x * w)
                index_tip_y = int(landmarks[8].y * h)
                
                # Invert X for mirroring (since we show mirrored webcam video)
                index_tip_x = w - index_tip_x
                cursor_pos = (index_tip_x, index_tip_y)
                
                # Determine Gesture
                if index_up and not middle_up and not ring_up and not pinky_up:
                    # ONLY Index finger up -> DRAW
                    gesture = 'draw'
                    # Visual feedback: Draw a green dot at cursor
                    cv2.circle(processed_frame, (int(landmarks[8].x * w), int(landmarks[8].y * h)), 10, (0, 255, 0), cv2.FILLED)
                elif index_up and middle_up and not ring_up and not pinky_up:
                    # Index and Middle up (Double fingers) -> SUBMIT
                    gesture = 'submit'
                    # Visual feedback: Draw a yellow circle in center
                    cv2.circle(processed_frame, (w // 2, h // 2), 15, (0, 255, 255), 2)
                elif index_up and middle_up and ring_up and pinky_up:
                    # Open palm (all fingers up) -> CLEAR / RESET
                    gesture = 'clear'
                    # Visual feedback: Draw a blue dot at cursor
                    cv2.circle(processed_frame, (int(landmarks[8].x * w), int(landmarks[8].y * h)), 10, (255, 0, 0), cv2.FILLED)
                else:
                    # Fist or other configurations -> NONE (No Operation)
                    gesture = 'none'
                    
                # Mirror the processed frame horizontally for standard webcam mirror experience
                processed_frame = cv2.flip(processed_frame, 1)
                
                # Since we mirrored, the cursor_pos needs to be aligned.
                # In the processing, we already mirrored index_tip_x.
                
                # Break after first hand to support only single hand interaction
                break
        else:
            # Mirror the frame even if no hand is detected
            processed_frame = cv2.flip(processed_frame, 1)
            
        return processed_frame, cursor_pos, gesture
