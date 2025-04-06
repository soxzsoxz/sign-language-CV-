import cv2
from ultralytics import YOLO

# โหลดโมเดล
model = YOLO('best.pt')

# เปิดวิดีโอ
cap = cv2.VideoCapture("images/test.mp4")

# ปรับขนาดวิดีโอที่ประมวลผลให้เล็กลง (ลดความละเอียด)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # กำหนดความกว้าง
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # กำหนดความสูง

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # ใช้โมเดล YOLO ประมวลผลเฟรม
    results = model(frame)
    
    # แสดงผลในแต่ละเฟรม
    results[0].show()  # ใช้ [0] เพื่อเรียกแสดงผลเฉพาะรายการแรก
    
    # กด 'q' เพื่อหยุดวิดีโอ
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
