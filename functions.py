import cv2
import numpy as np
import dlib

dnn_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Initialize dlib's facial landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

dnn_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Initialize dlib's facial landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_face(image, net=dnn_net, predictor=predictor):
    # Prepare the image for DNN face detection
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Convert bounding box to dlib rectangle format
            dlib_rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            landmarks = predictor(gray, dlib_rect)

            # Visualize landmarks
            # for p in landmarks.parts():
            #     cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), -1)

            # Get the bounding box for the face based on landmarks
            landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])
            x, y, w, h = cv2.boundingRect(landmarks_np)
            print(x,y,w,h)
            x -= 25
            y -= 25
            w += 50
            h += 50

            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            # Crop and resize the face
            face_crop = image[max(y-h//2,0):y+h, x:x+w]
            print(face_crop.shape)
            face_crop = cv2.resize(face_crop, (224, 224))
            return face_crop

    return None
