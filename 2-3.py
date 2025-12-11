import cv2
import os

template_path = "polyton.png"

#Проверка на существование файла
if not os.path.exists(template_path):
    print(f"Ошибка: файл '{template_path}' не найден!")
    print("Доступные файлы в папке:")
    for file in os.listdir("."):
        print(f"  - {file}")
    exit()

#Изображение метки
template = cv2.imread(template_path, 0)

if template is None:
    print(f"Не удалось загрузить '{template_path}'")
    print("Попробуйте:")
    print("1. Переименовать файл без пробелов: 'metka.png'")
    print("2. Проверить, что это действительно PNG файл")
    exit()

h, w = template.shape
print(f" Шаблон загружен: {template_path}")
print(f"  Размеры: {w}x{h} пикселей")

#Инициализируем камеру
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось открыть камеру!")
    exit()

print("\nКамера запущена. Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Ищем метку
    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    #Координаты метки
    x, y = -1, -1
    
    if max_val > 0.6: 
        x, y = max_loc
        bottom_right = (x + w, y + h)
        cv2.rectangle(frame, (x, y), bottom_right, (0, 255, 0), 2)
    
    cv2.putText(frame, f"X: {x}, Y: {y}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Match: {max_val:.2f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Отслеживание метки", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
