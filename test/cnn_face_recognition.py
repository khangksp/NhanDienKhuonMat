import cv2
import numpy as np
import tensorflow as tf
import json
import time
from tensorflow.keras.models import load_model

# Ẩn log không cần thiết của TensorFlow
tf.get_logger().setLevel('ERROR')

# Load mô hình CNN
model = load_model("models/cnn_face_recognition.h5")

# Load bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load ánh xạ ID -> Mã sinh viên
with open("labels.json", "r", encoding="utf-8") as f:
    id_to_name = json.load(f)

# Định kích thước ảnh đầu vào
IMG_SIZE = (66, 66)
CONFIDENCE_THRESHOLD = 0.7  # Ngưỡng tin cậy để xác định người quen

# Biến để kiểm soát hiển thị thông tin trong 3 giây
last_detected = None
display_until = 0

# Khởi động webcam
cap = cv2.VideoCapture(0)
print("🔍 Đang chạy nhận diện khuôn mặt...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi đọc camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    detected = False

    for (x, y, w, h) in faces:
        face_roi = cv2.resize(gray[y:y+h, x:x+w], IMG_SIZE)
        face_roi = np.expand_dims(face_roi, axis=-1)  # Thêm kênh màu xám
        face_roi = np.expand_dims(face_roi, axis=0)  # Thêm batch dimension
        face_roi = face_roi / 255.0  # Chuẩn hóa dữ liệu
        
        # Dự đoán nhãn và độ tin cậy
        predictions = model.predict(face_roi, verbose=0)  
        predicted_label = np.argmax(predictions)
        confidence = np.max(predictions)  # Xác suất cao nhất của dự đoán
        
        # Kiểm tra nếu độ tin cậy đủ cao
        if confidence >= CONFIDENCE_THRESHOLD:
            student_id = id_to_name.get(str(predicted_label), None)
        else:
            student_id = None  # Không xác định người lạ

        if student_id:
            detected = True
            if student_id != last_detected:
                last_detected = student_id
                display_until = time.time() + 3  # Hiển thị trong 3 giây
            
            if time.time() < display_until:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, student_id, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Nếu không có khuôn mặt được nhận diện, reset hiển thị
    if not detected:
        last_detected = None

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
