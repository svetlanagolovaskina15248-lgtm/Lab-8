import cv2
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
        match_value: величина совпадения шаблона с кадром
    """
    # Переводим кадр в оттенки серого
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Размеры кадра
    frame_h, frame_w = frame_gray.shape

    # Переменные для хранения лучшего результата
    best_value = -1
    best_top_left = (0, 0)
    best_bottom_right = (0, 0)

    # Набор масштабов метки
    scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    # Перебираем масштабы
    for scale in scales:
        resized_marker = cv2.resize(
            marker_gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR
        )

        marker_h, marker_w = resized_marker.shape

        # Если увеличенная метка больше кадра — пропускаем
        if marker_h >= frame_h or marker_w >= frame_w:
            continue

        # Выполняем поиск шаблона
        result = cv2.matchTemplate(frame_gray, resized_marker, cv2.TM_CCOEFF_NORMED)

        # Берём лучшее совпадение для текущего масштаба
        _, current_value, _, max_loc = cv2.minMaxLoc(result)

        # Если текущее совпадение лучше предыдущего — сохраняем его
        if current_value > best_value:
            best_value = current_value
            best_top_left = max_loc
            best_bottom_right = (max_loc[0] + marker_w, max_loc[1] + marker_h)

    # Проверяем, достаточно ли хорошее совпадение
    found = best_value >= threshold

    return found, best_top_left, best_bottom_right, best_value


# ============================================================
# ФУНКЦИЯ: отображение результата с координатами
# ============================================================
def draw_tracking_result_with_coordinates(frame, found, top_left, bottom_right, match_value):
    """
    Отображает результат поиска метки:
    - рамку вокруг метки
    - координаты метки в левом верхнем углу окна
    - коэффициент совпадения

    Параметры:
        frame: кадр с камеры
        found: найдена ли метка
        top_left: координаты левого верхнего угла метки
        bottom_right: координаты правого нижнего угла метки
        match_value: коэффициент совпадения
    """
    if found:
        # Рисуем рамку вокруг найденной метки
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Формируем текст с координатами метки
        coords_text = f"Coords: x={top_left[0]}, y={top_left[1]}"

        # Выводим координаты в левом верхнем углу экрана
        cv2.putText(
            frame,
            coords_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Выводим коэффициент совпадения
        match_text = f"Match: {match_value:.3f}"
        cv2.putText(
            frame,
            match_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
    else:
        # Если метка не найдена
        cv2.putText(
            frame,
            "Marker not found",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # Даже если не найдена, всё равно показываем коэффициент совпадения
        cv2.putText(
            frame,
            f"Match: {match_value:.3f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ ПРОГРАММЫ
# ============================================================
def main():
    """
    Основная логика программы:
    1. Загружаем изображение метки
    2. Открываем камеру
    3. Захватываем видеопоток
    4. Ищем метку на каждом кадре
    5. Выводим координаты метки в левом верхнем углу окна
    """
    # Формируем путь к метке относительно файла программы
    base_dir = Path(__file__).resolve().parent
    marker_path = str(base_dir / "images" / "variant-1.jpg")
    # Если файл PNG, замените последнюю часть пути на "variant-1.png"

    # Загружаем метку
    marker_gray = load_marker(marker_path)

    # Открываем камеру
    cap = open_camera()

    if cap is None:
        print("Ошибка: не удалось открыть камеру.")
        return

    print("Программа запущена.")
    print("Покажи метку перед камерой.")
    print("Для выхода нажмите клавишу q или ESC.")

    # Основной цикл чтения кадров
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Ошибка: не удалось получить кадр с камеры.")
            break

        # Ищем метку на кадре
        found, top_left, bottom_right, match_value = find_marker(frame, marker_gray)

        # Отображаем результат
        draw_tracking_result_with_coordinates(
            frame,
            found,
            top_left,
            bottom_right,
            match_value
        )

        # Показываем окно программы
        cv2.imshow("Marker Tracking With Coordinates", frame)

        # Завершение работы по нажатию q или ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Освобождаем камеру и закрываем окна
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