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

    # Проверяем, удалось ли загрузить файл
    if marker is None:
        raise FileNotFoundError(f"Не удалось открыть изображение метки: {marker_path}")

    # Переводим метку в полутоновый формат
    marker_gray = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)

    return marker_gray


# ============================================================
# ФУНКЦИЯ: открытие камеры
# ============================================================
def open_camera():
    """
    Пытается открыть доступную камеру.
    Для macOS последовательно проверяются индексы 0, 1 и 2.

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
# ФУНКЦИЯ: поиск метки на кадре
# ============================================================
def find_marker_multiscale(frame, marker_gray, threshold=0.35):
    """
    Ищет метку на кадре методом сопоставления шаблона на нескольких масштабах.

    Параметры:
        frame: текущий кадр с камеры
        marker_gray: полутоновое изображение метки
        threshold: минимальный уровень совпадения

    Возвращает:
        found: найдена ли метка
        top_left: координаты левого верхнего угла найденной области
        bottom_right: координаты правого нижнего угла найденной области
        best_val: лучший коэффициент совпадения
    """
    # Переводим кадр в полутоновый формат
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Размеры кадра
    frame_h, frame_w = frame_gray.shape

    # Здесь будут храниться лучшие результаты поиска
    best_val = -1
    best_top_left = (0, 0)
    best_bottom_right = (0, 0)

    # Проверяем несколько масштабов метки.
    # Это нужно, чтобы метка находилась, даже если она на камере чуть больше или меньше.
    scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    for scale in scales:
        # Изменяем размер метки
        resized_marker = cv2.resize(
            marker_gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR
        )

        marker_h, marker_w = resized_marker.shape

        # Пропускаем масштаб, если метка стала больше кадра
        if marker_h >= frame_h or marker_w >= frame_w:
            continue

        # Ищем шаблон на кадре
        result = cv2.matchTemplate(frame_gray, resized_marker, cv2.TM_CCOEFF_NORMED)

        # Получаем наилучшее совпадение для данного масштаба
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Если текущее совпадение лучше предыдущего — запоминаем его
        if max_val > best_val:
            best_val = max_val
            best_top_left = max_loc
            best_bottom_right = (max_loc[0] + marker_w, max_loc[1] + marker_h)

    # Проверяем, достаточно ли хорошее совпадение
    found = best_val >= threshold

    return found, best_top_left, best_bottom_right, best_val


# ============================================================
# ФУНКЦИЯ: отображение результата
# ============================================================
def draw_tracking_result(frame, found, top_left, bottom_right, match_value):
    """
    Отображает результат поиска метки на кадре.

    Параметры:
        frame: кадр с камеры
        found: найдена ли метка
        top_left: левый верхний угол найденной области
        bottom_right: правый нижний угол найденной области
        match_value: коэффициент совпадения
    """
    if found:
        # Рисуем рамку вокруг найденной метки
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Выводим сообщение об успешном обнаружении
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
        # Если метка не найдена
        cv2.putText(
            frame,
            "Marker not found",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    # Дополнительно выводим коэффициент совпадения
    cv2.putText(
        frame,
        f"Match: {match_value:.3f}",
        (10, 65),
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
    3. Считываем кадры в цикле
    4. Выполняем поиск метки
    5. Отображаем результат
    """
    # Автоматически строим путь к файлу метки относительно текущего .py файла
    base_dir = Path(__file__).resolve().parent
    marker_path = str(base_dir / "images" / "variant-1.jpg")
    # Если у тебя PNG, замени "variant-1.jpg" на "variant-1.png"

    # Загружаем изображение метки
    marker_gray = load_marker(marker_path)

    # Открываем камеру
    cap = open_camera()

    if cap is None:
        print("Ошибка: не удалось открыть камеру.")
        return

    print("Программа запущена.")
    print("Покажи метку прямо перед камерой.")
    print("Для выхода нажми q или ESC.")

    # Основной цикл чтения кадров
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Ошибка: не удалось получить кадр с камеры.")
            break

        # Ищем метку на текущем кадре
        found, top_left, bottom_right, match_value = find_marker_multiscale(
            frame,
            marker_gray,
            threshold=0.35
        )

        # Рисуем результат на кадре
        draw_tracking_result(frame, found, top_left, bottom_right, match_value)

        # Показываем окно программы
        cv2.imshow("Marker Tracking", frame)

        # Выход по q или ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Освобождаем ресурсы
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