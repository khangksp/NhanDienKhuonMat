import cv2
import pickle
import numpy as np
from skimage.feature import hog

# Load mô hình KNN đã train
with open("faceRe/models/knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

# Load bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Định kích thước ảnh đầu vào giống lúc train
IMG_SIZE = (66, 66)

def extract_hog_features(image):
    """ Trích xuất đặc trưng HOG từ ảnh xám """
    return hog(image, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm='L2-Hys')

def adjust_gamma(image, gamma=1.5):
    """ Điều chỉnh độ sáng bằng gamma correction """
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Khởi động webcam
cap = cv2.VideoCapture(0)

print("Đang chạy nhận diện khuôn mặt...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi đọc camera.")
        break

    # Tăng độ sáng trước khi xử lý
    frame = adjust_gamma(frame, gamma=1.5)  # Giá trị có thể thay đổi (1.5 - 1.8)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = cv2.resize(gray[y:y+h, x:x+w], IMG_SIZE)  # Resize về đúng kích thước train
        face_hog = extract_hog_features(face_roi).reshape(1, -1)  # Trích xuất HOG & reshape

        student_id = knn.predict(face_hog)[0]  # Dự đoán bằng KNN

        # Hiển thị kết quả trên khung hình
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {student_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
