import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Định dạng ảnh
IMG_SIZE = (66, 66)  # Kích thước ảnh
DATA_DIR = "images/"  # Thư mục chứa ảnh
MODEL_PATH = "models/cnn_face_recognition.h5"

# Đọc dữ liệu ảnh
X, y, labels = [], [], {}
label_index = 0
for person in os.listdir(DATA_DIR):
    person_path = os.path.join(DATA_DIR, person)
    if os.path.isdir(person_path):
        labels[label_index] = person  # Mapping index -> tên
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            X.append(img)
            y.append(label_index)
        label_index += 1

# Chuyển đổi thành numpy array
X = np.array(X).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1) / 255.0  # Chuẩn hóa về 0-1
y = to_categorical(np.array(y))  # One-hot encoding

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

# Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Lưu mô hình
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
np.save("models/labels.npy", labels)  # Lưu nhãn lớp

print(f"✅ Mô hình đã được lưu tại {MODEL_PATH}")
