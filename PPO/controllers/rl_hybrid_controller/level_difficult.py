import numpy as np


def choice_ray(front_state):
    """
    Разбивает массив расстояний лидара на 3 сектора: левый, центральный, правый.

    Args:
        front_state: список расстояний от лидара (обычно 360 значений)

    Returns:
        Кортеж из 3 списков: (danger_rays_left, danger_rays_front, danger_rays_right)
        Каждый список содержит расстояния < 15 метров в своём секторе
    """
    danger_rays_left = []
    danger_rays_front = []
    danger_rays_right = []

    for index, distance in enumerate(front_state):
        # Только близкие объекты (< 15м)
        if distance < 15:
            # Левый сектор: первые 1/3 лучей
            if index <= len(front_state) / 3:
                danger_rays_left.append(distance)
            # Центральный сектор: средние 1/3 лучей
            elif len(front_state) / 3 < index <= 2 * len(front_state) / 3:
                danger_rays_front.append(distance)
            # Правый сектор: последние 1/3 лучей
            else:
                danger_rays_right.append(distance)

    return danger_rays_left, danger_rays_front, danger_rays_right


def check_wheel_angle(wheel_angle):
    """
    Классифицирует угол поворота колёс для корректировки опасности.

    Args:
        wheel_angle: угол поворота колёс в радианах

    Returns:
        2.0 - поворот налево (< -0.2 рад)
        1.0 - прямо (-0.2 до 0.2 рад)
        0.0 - поворот направо (> 0.2 рад)
    """
    if wheel_angle < -0.2:
        return 2.0  # Налево
    elif -0.2 <= wheel_angle <= 0.2:
        return 1.0  # Прямо
    else:
        return 0.0  # Направо


def danger_level(danger_rays_left, danger_rays_front, danger_rays_right, speed, wheel_angle):
    """
    Вычисляет уровень опасности [0-1] для каждого сектора.

    Учитывает:
    - Расстояние до ближайшего объекта (чем ближе, тем опаснее)
    - Количество лучей с препятствиями (ширина объекта)
    - Скорость машины (быстрее = опаснее)
    - Направление движения (поворот в сторону объекта = опаснее)

    Args:
        danger_rays_*: списки расстояний в каждом секторе
        speed: текущая скорость
        wheel_angle: угол поворота колёс

    Returns:
        Список [danger_left, danger_front, danger_right], каждое значение [0-1]
    """
    rays = [danger_rays_left, danger_rays_front, danger_rays_right]

    # Нормализация скорости (0.1 - минимум, 40 - максимум)
    speed_norm = speed / 40.0 if speed > 0 else 0.1

    # Классификация направления
    wheel_factor_base = check_wheel_angle(wheel_angle)

    result = []

    for sector_index, sector_rays in enumerate(rays):
        # Минимальное расстояние в секторе (нормализованное)
        if len(sector_rays) > 0:
            min_distance = min(sector_rays)
            # Защита от нуля и нормализация (15м = 1.0, 0м = 0.0)
            min_rays = max(min_distance, 0.1) / 15.0
        else:
            min_rays = 1.0  # Нет препятствий = безопасно

        # Защита от деления на ноль
        if min_rays < 0.01:
            min_rays = 0.01

        # Корректировка по направлению движения
        # Если едем прямо (1.0) - фронтальная опасность важнее
        # Если поворачиваем - боковая опасность с соответствующей стороны важнее
        if sector_index == 1:  # Фронт
            # wheel_factor_base: 2=влево, 1=прямо, 0=вправо
            # Для фронта: если едем прямо (1), множитель 1.0
            # Если влево (2), множитель 1.5 (смотрим направо меньше)
            # Если вправо (0), множитель 1.5 (смотрим налево меньше)
            wheel_factor = abs((1.0 + wheel_factor_base) / 2.0)
        else:  # Боковые секторы
            # Если поворачиваем в сторону сектора - опасность выше
            wheel_factor = abs((1.0 - wheel_factor_base) / 2.0)

        # Формула опасности:
        # - Больше лучей = шире объект
        # - Выше скорость = меньше времени на реакцию
        # - Меньше расстояние = ближе объект
        # - Множитель направления
        danger_value = (len(sector_rays) / 19.0) * (speed_norm / min_rays) * wheel_factor

        # Нормализация в [0, 1]
        norm_danger = float(np.clip(danger_value, 0, 1))
        result.append(norm_danger)

    return result


class CurriculumManager:
    """
    Управляет постепенным усложнением задачи (Curriculum Learning).

    При достижении 70% успеха на текущем уровне в течение 50+ эпизодов,
    препятствие сдвигается ближе к цели.
    """

    def __init__(self):
        self.level = 0  # Текущий уровень сложности
        self.success_threshold = 0.7  # Порог успеха для повышения (70%)
        self.episodes_at_level = 0  # Эпизодов на текущем уровне
        self.success_count = 0  # Успешных эпизодов всего
        self.total_episodes = 0  # Всего эпизодов на уровне

    def update(self, success):
        """
        Обновляет статистику и проверяет повышение уровня.

        Args:
            success: bool, достигнута ли цель в эпизоде

        Returns:
            True если уровень повышен, иначе False
        """
        self.total_episodes += 1
        self.episodes_at_level += 1

        if success:
            self.success_count += 1

        # Проверяем условие повышения
        if self.total_episodes == 0:
            return False

        success_rate = self.success_count / self.total_episodes

        # Повышаем уровень при хорошем успехе и достаточном опыте
        if success_rate > self.success_threshold and self.episodes_at_level > 50:
            self.level += 1
            self.episodes_at_level = 0
            self.success_count = 0
            self.total_episodes = 0
            print(f"🎓 УРОВЕНЬ ПОВЫШЕН ДО {self.level}!")
            print(f"   Новая позиция препятствия: {self.get_obstacle_position()}")
            return True

        return False

    def get_obstacle_position(self):
        """
        Возвращает позицию препятствия для текущего уровня.

        Уровень 0: x=0   (центр)
        Уровень 1: x=2   (ближе к цели)
        Уровень 2: x=4
        Уровень 3: x=6
        Уровень 4+: x=8  (максимум, близко к цели)

        Returns:
            [x, y, z] позиция препятствия
        """
        base_x = 0.0
        # Сдвиг на 2 метра за уровень, максимум +8
        offset = min(self.level * 2.0, 8.0)
        return [base_x + offset, 0.5, 0.0]

    def get_obstacle_size(self):
        """
        Возвращает размер препятствия для текущего уровня.

        Уровень 0: 1x1x1 м
        Каждый уровень: +0.3 м, максимум 2.5 м

        Returns:
            [size_x, size_y, size_z]
        """
        size = 1.0 + min(self.level * 0.3, 1.5)
        return [size, size, size]

    def get_stats(self):
        """Возвращает строку со статистикой для логирования."""
        success_rate = 0
        if self.total_episodes > 0:
            success_rate = 100 * self.success_count / self.total_episodes
        return f"Level:{self.level} | Success:{success_rate:.1f}% ({self.success_count}/{self.total_episodes})"

    def reset_level(self):
        """Сброс к начальному уровню (для нового обучения)."""
        self.level = 0
        self.episodes_at_level = 0
        self.success_count = 0
        self.total_episodes = 0
        print("🔄 Curriculum сброшен к уровню 0")