import cv2
import pickle
import numpy as np
from django.http import StreamingHttpResponse
from django.shortcuts import render
from skimage.feature import hog

# Load mô hình KNN đã train
with open("models/knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

# Load bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

IMG_SIZE = (66, 66)  # Kích thước ảnh giống lúc train

def extract_hog_features(image):
    """ Trích xuất đặc trưng HOG từ ảnh xám """
    return hog(image, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm='L2-Hys')

def video_stream():
    cap = cv2.VideoCapture(0)  # Mở camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], IMG_SIZE)  # Resize ảnh
            face_hog = extract_hog_features(face_roi).reshape(1, -1)  # Trích xuất HOG

            student_id = knn.predict(face_hog)[0]  # Dự đoán bằng KNN

            # Vẽ khung nhận diện và hiển thị ID
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {student_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def video_feed(request):
    return StreamingHttpResponse(video_stream(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def home(request):
    return render(request, 'recognition/home.html')
