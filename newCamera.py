import cv2
import pyautogui as pag
import math
import numpy as np
import mediapipe as mp

class myCamera:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.cap = None
        self.width, self.height, self.channels = 0, 0, 0

    def setupCamera(self, stream = False): 
        if (stream): 
            self.cap = cv2.VideoCapture('tcp://192.168.1.20:3000')
            return
        self.cap = cv2.VideoCapture(0)

    def runServices(self):
        with self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())

                
                lmList = []
                if results.multi_hand_landmarks:
                    myHand = results.multi_hand_landmarks[0]
                    for id, lm in enumerate(myHand.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])          

                self.height, self.width, self.channels = image.shape
                self.center_x = int(self.width / 2)
                self.center_y = int(self.height / 2)
                cv2.circle(image, (int(self.width / 2), int(self.height / 2)), 10, (0,0,222))
                # Assigning variables for Thumb and Index finger position
                if len(lmList) != 0:
                    index_x, index_y = lmList[8][1], lmList[8][2]

                    cv2.circle(image, (index_x,index_y), 15, (0,0,255)) # Index finger

                    

                    distance_from_center = math.hypot(index_x - self.center_x, index_y - self.center_y)
                    cv2.line(image, (index_x, index_y), (self.center_x, self.center_y), (255, 255, 0), 1)
                    print(distance_from_center)


                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('Hand tracking', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        self.cap.release()
 
if __name__ == "__main__":
    camera = myCamera()
    camera.setupCamera()
    camera.runServices()