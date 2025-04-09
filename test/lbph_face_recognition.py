import cv2
import numpy as np
import mysql.connector
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Kết nối MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",  # Thay user MySQL của bạn
    password="123456",  # Thay password của bạn
    database="face_recognition"
)
cursor = conn.cursor()

# Kiểm tra trainer.yml
if not os.path.exists("models/lbph_trainer.yml"):
    print("Lỗi: Không tìm thấy file trainer.yml. Vui lòng train model trước!")
    exit()

# Load model đã train
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/lbph_trainer.yml")

# Load bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể mở webcam!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        # Dự đoán danh tính
        student_id, confidence = recognizer.predict(face_roi)

        if confidence < 65:  
            cursor.execute("SELECT name FROM students WHERE student_id = %s", (str(student_id),))
            result = cursor.fetchone()

            if result:
                name = result[0]
                label = f"{name} ({student_id})"
                color = (0, 255, 0)  # Xanh lá cây nếu nhận diện được
            else:
                label = "Không tìm thấy trong CSDL"
                color = (0, 255, 255)  # Vàng nếu không có trong database
        else:
            label = "Không xác định"
            color = (0, 0, 255)  # Đỏ nếu không nhận diện được

        # Hiển thị thông tin lên khung hình
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()
