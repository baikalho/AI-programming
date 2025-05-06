import cv2
from ultralytics import YOLO

# 모델 로드 (YOLOv5s 또는 YOLOv8n 등, ultralytics 라이브러리 사용)
model = YOLO("yolov8n.pt")  # 가볍고 빠른 모델

# 이미지 로드
image_path = 'sample4.jpg'  # 이미지 파일 경로
image = cv2.imread(image_path)

# 모델 추론
results = model(image)

# 결과에서 바운딩 박스 정보 추출
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌상단, 우하단 좌표
        conf = float(box.conf[0])  # confidence score
        cls_id = int(box.cls[0])   # class ID
        label = model.names[cls_id]  # 클래스 이름

        # 사각형과 라벨 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# 윈도우 이름 지정
window_name = "Detected Objects"

# 윈도우 생성 (크기 조절 가능하도록 설정)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# 원하는 사이즈 설정 (예: 800x600)
cv2.resizeWindow(window_name, 800, 600)

# 이미지 출력
cv2.imshow(window_name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지 화면에 표시
#cv2.imshow("Detected Objects", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
