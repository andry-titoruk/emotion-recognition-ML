import cv2

# 1. Завантажуємо вбудований навчений класифікатор для облич
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Підключаємося до веб-камери
cap = cv2.VideoCapture(0)

print("Натисніть 'q' у вікні з камерою, щоб вийти")

while True:
    # Зчитуємо кадр
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Перетворюємо зображення в сіре (алгоритму легше працювати без кольору)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4. Шукаємо обличчя на кадрі
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 5. Малюємо зелений квадрат навколо кожного знайденого обличчя
    for (x, y, w, h) in faces:
        # (x, y) - верхній лівий кут, (x+w, y+h) - нижній правий
        # (0, 255, 0) - колір BGR (зелений), 2 - товщина лінії
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Вирізаємо область обличчя (ROI - Region of Interest)
        roi_gray = gray[y:y + h, x:x + w]
        # Змінюємо розмір до 48x48 для майбутньої нейромережі
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    # Показуємо результат
    cv2.imshow('Face Detection Test', frame)

    # Вихід при натисканні 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Звільняємо ресурси
cap.release()
cv2.destroyAllWindows()
