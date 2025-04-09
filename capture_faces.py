import cv2
import os
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Kiểm tra nếu đã có thông tin sinh viên từ lần chạy trước
try:
    with open("last_user.txt", "r") as file:
        last_username, last_student_id = file.read().splitlines()
except FileNotFoundError:
    last_username, last_student_id = None, None

# Hỏi người dùng có muốn dùng lại thông tin cũ không
if last_username and last_student_id:
    print(f"\n🔹 Lần trước bạn nhập: {last_username} - {last_student_id}")
    reuse = input("Bạn có muốn dùng lại thông tin này không? (y/n): ").strip().lower()
else:
    reuse = "n"

if reuse == "y":
    username, student_id = last_username, last_student_id
else:
    username = input("Nhập họ và tên sinh viên: ").strip().replace(" ", "_")
    student_id = input("Nhập mã số sinh viên: ").strip()
    # Lưu lại thông tin cho lần sau
    with open("last_user.txt", "w") as file:
        file.write(f"{username}\n{student_id}")

# Tạo thư mục lưu ảnh theo cấu trúc images/tên_sinh_viên/
output_folder = os.path.join("images", username)
os.makedirs(output_folder, exist_ok=True)

# Kiểm tra số ảnh hiện có để tiếp tục đánh số
existing_images = [f for f in os.listdir(output_folder) if f.startswith(student_id) and f.endswith(".jpg")]
count = len(existing_images) + 1  # Bắt đầu từ số tiếp theo

max_images = count + 20  # Chụp thêm 20 ảnh mới

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while count < max_images:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Thêm padding 20% để khuôn mặt lớn hơn
        pad = int(0.2 * w)
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + w + pad, frame.shape[1]), min(y + h + pad, frame.shape[0])
        
        face = frame[y1:y2, x1:x2]
        face_resized = cv2.resize(face, (500, 500))  # Resize về kích thước cố định
        
        # Lưu ảnh với tên mssv_1.jpg, mssv_2.jpg,...
        filename = os.path.join(output_folder, f"{student_id}_{count}.jpg")
        cv2.imwrite(filename, face_resized)
        count += 1

        # Vẽ khung xanh quanh khuôn mặt sau khi đã lưu
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Capturing Faces", frame)
    
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Đã lưu {max_images - len(existing_images)} ảnh vào thư mục {output_folder}.")