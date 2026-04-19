import cv2
import numpy as np
from pathlib import Path


# ============================================================
# ФУНКЦИЯ: загрузка изображения метки
# ============================================================
def load_marker(marker_path: str):
    """
    Загружает изображение метки и переводит его в полутоновый формат.

    Параметры:
        marker_path (str): путь к изображению метки

    Возвращает:
        marker_gray: полутоновое изображение метки
    """
    marker = cv2.imread(marker_path)

    if marker is None:
        raise FileNotFoundError(f"Не удалось открыть изображение метки: {marker_path}")

    marker_gray = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
    return marker_gray


# ============================================================
# ФУНКЦИЯ: загрузка изображения мухи
# ============================================================
def load_fly_image(fly_path: str):
    """
    Загружает изображение мухи с сохранением альфа-канала.
    Это важно для корректного наложения PNG с прозрачностью.

    Параметры:
        fly_path (str): путь к файлу fly64.png

    Возвращает:
        fly_image: изображение мухи
    """
    fly_image = cv2.imread(fly_path, cv2.IMREAD_UNCHANGED)

    if fly_image is None:
        raise FileNotFoundError(f"Не удалось открыть изображение мухи: {fly_path}")

    return fly_image


# ============================================================
# ФУНКЦИЯ: открытие камеры
# ============================================================
def open_camera():
    """
    Пытается открыть доступную камеру.
    Для macOS проверяются индексы 0, 1 и 2 без использования CAP_DSHOW.

    Возвращает:
        cap: объект видеозахвата
    """
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)

        if cap.isOpened():
            print(f"Камера открыта успешно. Индекс: {camera_index}")
            return cap

        cap.release()

    return None


# ============================================================
# ФУНКЦИЯ: поиск метки на нескольких масштабах
# ============================================================
def find_marker(frame, marker_gray, threshold=0.35):
    """
    Ищет метку на кадре методом сопоставления шаблона на нескольких масштабах.

    Параметры:
        frame: текущий кадр с камеры
        marker_gray: полутоновое изображение метки
        threshold: минимальный допустимый коэффициент совпадения

    Возвращает:
        found: найдена ли метка
        top_left: координаты левого верхнего угла метки
        bottom_right: координаты правого нижнего угла метки
        match_value: коэффициент совпадения
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_h, frame_w = frame_gray.shape

    best_value = -1
    best_top_left = (0, 0)
    best_bottom_right = (0, 0)

    scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    for scale in scales:
        resized_marker = cv2.resize(
            marker_gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR
        )

        h, w = resized_marker.shape

        if h >= frame_h or w >= frame_w:
            continue

        result = cv2.matchTemplate(frame_gray, resized_marker, cv2.TM_CCOEFF_NORMED)
        _, current_value, _, max_loc = cv2.minMaxLoc(result)

        if current_value > best_value:
            best_value = current_value
            best_top_left = max_loc
            best_bottom_right = (max_loc[0] + w, max_loc[1] + h)

    found = best_value >= threshold

    return found, best_top_left, best_bottom_right, best_value


# ============================================================
# ФУНКЦИЯ: наложение PNG с прозрачностью
# ============================================================
def overlay_png(background, overlay, x, y):
    """
    Накладывает PNG-изображение overlay на background в точку (x, y),
    где (x, y) — координаты левого верхнего угла overlay.

    Функция поддерживает альфа-канал PNG.

    Параметры:
        background: кадр камеры
        overlay: изображение мухи
        x, y: координаты левого верхнего угла мухи
    """
    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]

    # Если изображение полностью вне кадра, ничего не делаем
    if x >= bg_w or y >= bg_h or x + ov_w <= 0 or y + ov_h <= 0:
        return

    # Ограничиваем область наложения размерами кадра
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + ov_w, bg_w)
    y2 = min(y + ov_h, bg_h)

    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    background_crop = background[y1:y2, x1:x2]

    # Если у изображения есть альфа-канал
    if overlay_crop.shape[2] == 4:
        overlay_rgb = overlay_crop[:, :, :3]
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha = np.dstack((alpha, alpha, alpha))
    else:
        overlay_rgb = overlay_crop
        alpha = np.ones_like(overlay_rgb, dtype=np.float32)

    result = (overlay_rgb * alpha + background_crop * (1 - alpha)).astype(np.uint8)
    background[y1:y2, x1:x2] = result


# ============================================================
# ФУНКЦИЯ: наложение мухи в центр метки
# ============================================================
def put_fly_on_marker_center(frame, fly_image, top_left, bottom_right):
    """
    Накладывает изображение мухи так, чтобы центр мухи совпал с центром метки.

    Параметры:
        frame: текущий кадр
        fly_image: изображение мухи
        top_left: левый верхний угол метки
        bottom_right: правый нижний угол метки
    """
    # Центр найденной метки
    marker_center_x = (top_left[0] + bottom_right[0]) // 2
    marker_center_y = (top_left[1] + bottom_right[1]) // 2

    # Размеры изображения мухи
    fly_h, fly_w = fly_image.shape[:2]

    # Координаты верхнего левого угла мухи,
    # чтобы её центр совпал с центром метки
    fly_x = marker_center_x - fly_w // 2
    fly_y = marker_center_y - fly_h // 2

    overlay_png(frame, fly_image, fly_x, fly_y)


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ ПРОГРАММЫ
# ============================================================
def main():
    """
    Основная логика программы:
    1. Загружаем метку
    2. Загружаем изображение мухи
    3. Открываем камеру
    4. Ищем метку на каждом кадре
    5. Накладываем муху в центр метки
    """
    base_dir = Path(__file__).resolve().parent
    marker_path = str(base_dir / "images" / "variant-1.jpg")
    # Если файл PNG, замените последнюю часть пути на "variant-1.png"

    fly_path = str(base_dir / "images" / "fly64.png")

    marker_gray = load_marker(marker_path)
    fly_image = load_fly_image(fly_path)

    cap = open_camera()

    if cap is None:
        print("Ошибка: не удалось открыть камеру.")
        return

    print("Программа запущена.")
    print("Покажи метку перед камерой.")
    print("Для выхода нажмите клавишу q или ESC.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Ошибка: не удалось получить кадр с камеры.")
            break

        found, top_left, bottom_right, match_value = find_marker(frame, marker_gray)

        cv2.putText(
            frame,
            f"Match: {match_value:.3f}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        if found:
            # Рисуем рамку вокруг найденной метки
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            # Накладываем муху в центр метки
            put_fly_on_marker_center(frame, fly_image, top_left, bottom_right)

            # Показываем статус
            cv2.putText(
                frame,
                "Marker found",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "Marker not found",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        cv2.imshow("Fly on Marker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# ТОЧКА ВХОДА В ПРОГРАММУ
# ============================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Программа остановлена пользователем")