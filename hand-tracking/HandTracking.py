import cv2
import mediapipe as mp
import pyautogui as pag
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
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
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    
    lmList = []
    if results.multi_hand_landmarks:
      myHand = results.multi_hand_landmarks[0]
      for id, lm in enumerate(myHand.landmark):
        h, w, c = image.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])          

    # Assigning variables for Thumb and Index finger position
    if len(lmList) != 0:
      x_0, y_0 = lmList[8][1], lmList[8][2]
      x_1, y_1 = lmList[4][1], lmList[4][2]

      # Marking Index finger  
      cv2.circle(image, (x_0,y_0),15,(255,255,255)) 
      cv2.circle(image, (x_1,y_1),15,(255,255,255))

      ret,frame = cap.read()
      windowWidth=frame.shape[1]
      windowHeight=frame.shape[0]
      pag.moveTo((windowWidth - x_1), y_1)

      thumb_index_length = math.hypot(x_0-x_1,y_0-y_1)
      if thumb_index_length < 50:
        pag.click()


    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Hand tracking', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
