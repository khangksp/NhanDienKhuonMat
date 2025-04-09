import cv2
import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog

# Load bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Đường dẫn đến thư mục chứa ảnh
dataset_path = "images/"

# Dữ liệu train
X_train = []
y_train = []

print("📸 Đang trích xuất dữ liệu khuôn mặt...")

def extract_hog_features(image):
    """ Trích xuất đặc trưng HOG từ ảnh xám """
    return hog(image, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm='L2-Hys')

# Định kích thước ảnh đầu vào cố định
IMG_SIZE = (66, 66)  

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue  

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Lỗi đọc ảnh: {image_name}, bỏ qua.")
            continue  

        try:
            student_id = int(image_name.split('_')[0])
        except ValueError:
            print(f"Không thể lấy ID từ {image_name}, bỏ qua.")
            continue

        # Cân bằng Histogram để cải thiện độ tương phản
        img = cv2.equalizeHist(img)

        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = cv2.resize(img[y:y+h, x:x+w], IMG_SIZE)  # Resize về kích thước cố định
            features = extract_hog_features(face)  # Trích xuất đặc trưng HOG
            
            X_train.append(features)  
            y_train.append(student_id)

print(f"Đã thu thập {len(y_train)} khuôn mặt.")

# Chuyển dữ liệu về dạng numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Kích thước đầu vào HOG: {X_train.shape}")

# Huấn luyện mô hình KNN
print("🏋️‍♂️ Đang huấn luyện mô hình KNN...")
knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')  # Manhattan thường tốt hơn Euclidean với HOG
knn.fit(X_train, y_train)

# Lưu mô hình
if not os.path.exists("models"):
    os.makedirs("models")

with open("/faceRe/models/knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

print("Huấn luyện hoàn tất, Mô hình đã được lưu vào models/knn_model.pkl")
