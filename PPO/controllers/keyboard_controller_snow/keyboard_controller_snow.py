#!/usr/bin/env python3
"""
Keyboard controller with REAL snow/ice physics (как в коде друга)
- Снег: скорость зависит от расстояния до центра, формирование колеи, эффект заноса
- Лёд: сильное скольжение, потеря управления при поворотах
"""

from vehicle import Driver
from controller import Keyboard, Supervisor, Lidar, GPS
import math
import random
import numpy as np

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ
# =============================================================================
driver = Driver()
supervisor = Supervisor()
keyboard = Keyboard()
keyboard.enable(50)

# =============================================================================
# КОНСТАНТЫ
# =============================================================================
MAX_SPEED = 30.0
MAX_STEERING_ANGLE = 0.8

# Параметры столкновений
COLLISION_DISTANCE = 0.5
GOAL_DISTANCE = 4.0

# Параметры респауна
RESET_DELAY = 2.0
RESET_POSITION = [0.16, -45, 0]
RESET_ROTATION = [0, 1, 0, 0]

# Параметры физики снега (как в коде друга)
KOEF_SNOW = 0.97  # Базовый коэффициент замедления на снегу
KOEF_SNOW_SLIP = 0.8  # Коэффициент скольжения при заносе
TRACK_SIZE = 30  # Размер карты колеи (30x30)

# Параметры физики льда (отдельно от снега!)
# Лёд: машина скользит, нет колеи, другое замедление
ICE_BASE_SPEED = 0.85  # Базовое замедление на льду
ICE_SLIP_FACTOR = 0.75  # Фактор скольжения при поворотах
ICE_TURN_SLIP = 0.7  # Дополнительное скольжение при резких поворотах

# =============================================================================
# ПОИСК НОД
# =============================================================================
print("=" * 60)
print("ТЕСТИРОВАНИЕ ФИЗИКИ СНЕГА И ЛЬДА (как в коде друга)")
print("=" * 60)

# Поиск автомобиля
car_node = supervisor.getFromDef("CAR")
if car_node is None:
    print("⚠️ CAR не найден, используем getSelf()")
    car_node = driver.getSelf()
else:
    pos = car_node.getPosition()
    print(f"✅ CAR: найдена на X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f}")

# Поиск цели
target_node = supervisor.getFromDef("TARGET")
if target_node:
    target_pos = target_node.getPosition()
    print(f"✅ TARGET: найдена на X={target_pos[0]:.1f}, Y={target_pos[1]:.1f}, Z={target_pos[2]:.1f}")
    try:
        target_node.removeBoundingObject()
        target_node.enableContactPointsTracking(False)
        target_node.resetPhysics()
        print("   👻 Физика цели отключена")
    except Exception as e:
        print(f"   ⚠️ Не удалось отключить физику: {e}")
else:
    print("❌ TARGET: не найдена!")

# =============================================================================
# ПОИСК СНЕЖНЫХ ПОЛЕЙ И ЛЬДА
# =============================================================================
snow_fields = []  # Список нод снежных полей
snow_positions = []  # Позиции снежных полей [x, y, z]
ice_fields = []  # Список нод ледяных полей
ice_positions = []  # Позиции ледяных полей

root = supervisor.getRoot()
children = root.getField("children")

for i in range(children.getCount()):
    node = children.getMFNode(i)
    try:
        name_field = node.getField("name")
        if name_field:
            name = name_field.getSFString()
            if "snow" in name.lower():
                pos = node.getField("translation").getSFVec3f()
                snow_fields.append(node)
                snow_positions.append(pos)
                print(f"❄️ Снежное поле: '{name}' (X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f})")
            elif "ice" in name.lower():
                pos = node.getField("translation").getSFVec3f()
                ice_fields.append(node)
                ice_positions.append(pos)
                print(f"🧊 Ледяное поле: '{name}' (X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f})")
    except:
        pass

# =============================================================================
# ПОИСК ПРЕПЯТСТВИЙ
# =============================================================================
obstacle_nodes = []
for i in range(1, 6):
    obs = supervisor.getFromDef(f"OBSTACLE_{i}")
    if obs:
        obstacle_nodes.append(obs)
        pos = obs.getPosition()
        print(f"🔴 Препятствие {i}: X={pos[0]:.1f}, Y={pos[1]:.1f}")

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ СЕНСОРОВ
# =============================================================================
lidar_front = driver.getDevice("lidar_front")
lidar_left = driver.getDevice("lidar_left")
lidar_right = driver.getDevice("lidar_right")
gps = driver.getDevice("gps")
imu = driver.getDevice("imu")

for device, name in [(lidar_front, "front"), (lidar_left, "left"),
                     (lidar_right, "right"), (gps, "GPS"), (imu, "IMU")]:
    if device:
        device.enable(50)
        print(f"✅ {name}: активирован")

# =============================================================================
# ПЕРЕМЕННЫЕ ДЛЯ ФИЗИКИ СНЕГА (как в коде друга)
# =============================================================================
# Карта колеи для каждого снежного поля (30x30)
track_maps = []
for _ in range(len(snow_fields)):
    track_maps.append([[0.0 for _ in range(TRACK_SIZE)] for _ in range(TRACK_SIZE)])

# Для отслеживания скольжения
check_time = 0
check_slip = False


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================
def get_car_position():
    """Получение позиции машины: [X, Y, Z]"""
    if gps:
        vals = gps.getValues()
        return [float(vals[0]), float(vals[1]), float(vals[2])]
    elif car_node:
        return list(car_node.getPosition())
    return [0, 0, 0]


def get_yaw():
    """Получение угла поворота машины."""
    if car_node:
        try:
            orientation = car_node.getOrientation()
            yaw = math.atan2(orientation[2], orientation[8])
            return yaw
        except:
            pass
    return 0.0


def get_lidar_min_distance(lidar):
    if not lidar:
        return float('inf')
    data = lidar.getRangeImage()
    if not data or len(data) == 0:
        return float('inf')
    valid = [x for x in data if 0.1 < x < 100]
    return min(valid) if valid else float('inf')


def check_goal_3d():
    if target_node is None:
        return False, float('inf')
    car_pos = get_car_position()
    target_pos = target_node.getPosition()
    dx = car_pos[0] - target_pos[0]
    dy = car_pos[1] - target_pos[1]
    dz = car_pos[2] - target_pos[2]
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    return distance < GOAL_DISTANCE, distance


def check_collisions_lidar():
    front_dist = get_lidar_min_distance(lidar_front)
    left_dist = get_lidar_min_distance(lidar_left)
    right_dist = get_lidar_min_distance(lidar_right)
    min_dist = min(front_dist, left_dist, right_dist)

    if front_dist < COLLISION_DISTANCE:
        return True, f"FRONT lidar={front_dist:.2f}m", min_dist
    elif left_dist < COLLISION_DISTANCE:
        return True, f"LEFT lidar={left_dist:.2f}m", min_dist
    elif right_dist < COLLISION_DISTANCE:
        return True, f"RIGHT lidar={right_dist:.2f}m", min_dist
    return False, None, min_dist


def get_half_size(z):
    """Вычисление размера снежного поля на основе высоты (как в коде друга)."""
    z = max(-3.2, min(0, z))
    return 15 * (z / -3.2)


def is_in_snow(car_pos, snow_index):
    """Проверка нахождения в конкретном снежном поле."""
    if snow_index >= len(snow_positions):
        return None, None, None

    snow_pos = snow_positions[snow_index]
    snow_z = snow_pos[2] if len(snow_pos) > 2 else -2.0

    HALF_SIZE = get_half_size(snow_z)
    if HALF_SIZE < 1:
        return None, None, None

    dx = car_pos[0] - snow_pos[0]
    dy = car_pos[1] - snow_pos[1]

    if abs(dx) < HALF_SIZE and abs(dy) < HALF_SIZE:
        distance = math.sqrt(dx * dx + dy * dy)
        max_dist = HALF_SIZE
        return distance, max_dist, snow_index
    return None, None, None


def apply_snow_physics(car_pos, current_speed, current_steering):
    """
    Применение физики снега (как в коде друга).
    Возвращает новую скорость.
    """
    global check_time, check_slip

    final_speed = current_speed

    for i in range(len(snow_fields)):
        dist, max_dist, idx = is_in_snow(car_pos, i)
        if dist is not None:
            # 1. Базовое замедление на снегу
            if max_dist > 0:
                final_speed = current_speed * min((KOEF_SNOW + 0.092 * dist / max_dist), 1)

            # 2. Учёт колеи (проторенный путь)
            snow_pos = snow_positions[i]
            local_x = int((car_pos[0] - snow_pos[0] + TRACK_SIZE / 2))
            local_y = int((car_pos[1] - snow_pos[1] + TRACK_SIZE / 2))

            if 0 <= local_x < TRACK_SIZE and 0 <= local_y < TRACK_SIZE:
                track_value = track_maps[i][local_x][local_y]
                # Колея даёт бонус к скорости (как в коде друга: + track_value*3/100)
                final_speed *= (1 + track_value * 0.03)
                final_speed = min(final_speed, current_speed)

            # 3. Эффект заноса на снегу (как в коде друга)
            if check_slip or check_time != 0:
                final_speed *= KOEF_SNOW_SLIP
                check_time += 1
                if check_time > 10:
                    check_slip = False
                    check_time = 0

            # 4. Занос при резких поворотах
            if abs(current_steering) > 0.3:
                check_slip = True
                check_time = 1

            break

    return final_speed


def update_track(car_pos, current_speed, current_steering):
    """Обновление карты колеи (как в коде друга)."""
    for i in range(len(snow_fields)):
        dist, max_dist, idx = is_in_snow(car_pos, i)
        if dist is not None:
            snow_pos = snow_positions[i]

            local_x = int((car_pos[0] - snow_pos[0] + TRACK_SIZE / 2))
            local_y = int((car_pos[1] - snow_pos[1] + TRACK_SIZE / 2))

            if 0 <= local_x < TRACK_SIZE and 0 <= local_y < TRACK_SIZE:
                radius = 2
                speed_val = abs(current_speed)

                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        nx, ny = local_x + dx, local_y + dy
                        if 0 <= nx < TRACK_SIZE and 0 <= ny < TRACK_SIZE:
                            if speed_val > 1:
                                track_maps[i][nx][ny] += 0.0002
                                track_maps[i][nx][ny] = min(1.0, track_maps[i][nx][ny])
                            else:
                                track_maps[i][nx][ny] -= 0.0002
                                track_maps[i][nx][ny] = max(0.0, track_maps[i][nx][ny])
            break


def is_in_ice(car_pos):
    """Проверка нахождения на льду."""
    for i, ice_pos in enumerate(ice_positions):
        if (abs(car_pos[0] - ice_pos[0]) < 18 and
                abs(car_pos[1] - ice_pos[1]) < 18):
            return True
    return False


def apply_ice_physics(car_pos, current_speed, current_steering):
    """
    Применение физики льда (отдельно от снега!).
    Лёд: сильное скольжение, потеря управления.
    """
    if not is_in_ice(car_pos):
        return current_speed

    # Базовое замедление на льду
    final_speed = current_speed * ICE_BASE_SPEED

    # Сильное скольжение при поворотах (потеря управления)
    if abs(current_steering) > 0.15:
        # Чем сильнее поворот, тем больше скольжение
        slip_amount = abs(current_steering) / MAX_STEERING_ANGLE
        final_speed *= ICE_SLIP_FACTOR * (1 - slip_amount * 0.3)

        # Дополнительное скольжение при резких поворотах
        if abs(current_steering) > 0.4:
            final_speed *= ICE_TURN_SLIP

    # При высокой скорости скольжение усиливается
    if abs(current_speed) > 15:
        final_speed *= 0.9

    return final_speed


def randomize_obstacles():
    """Случайная перегенерация препятствий."""
    print("\n  🎲 Перегенерация препятствий...")

    car_pos = get_car_position()
    min_distance_from_car = 8.0

    for i, obs in enumerate(obstacle_nodes):
        if not obs:
            continue

        for attempt in range(30):
            new_x = random.uniform(-40, 40)
            new_y = random.uniform(-35, 35)
            new_z = 0.50

            dist_to_car = math.sqrt((new_x - car_pos[0]) ** 2 + (new_y - car_pos[1]) ** 2)
            if dist_to_car < min_distance_from_car:
                continue

            valid = True
            for j, other_obs in enumerate(obstacle_nodes):
                if j >= i:
                    break
                if other_obs:
                    try:
                        other_pos = other_obs.getPosition()
                        dist = math.sqrt((new_x - other_pos[0]) ** 2 + (new_y - other_pos[1]) ** 2)
                        if dist < 5.0:
                            valid = False
                            break
                    except:
                        pass

            if valid:
                try:
                    obs.getField("translation").setSFVec3f([new_x, new_y, new_z])
                    obs.resetPhysics()
                    print(f"    Препятствие {i + 1}: X={new_x:5.1f}, Y={new_y:5.1f}")
                except:
                    pass
                break


def reset_car(reason="RESET"):
    """Сброс машины на стартовую позицию."""
    global in_reset, collision_detected, goal_reached, current_speed, current_steering
    global check_time, check_slip

    if reason == "GOAL":
        header = "🎯 GOAL REACHED! 🎯"
    elif reason == "COLLISION":
        header = "💥 COLLISION! 💥"
    else:
        header = "🔄 РУЧНОЙ СБРОС"

    print(f"\n{'=' * 60}")
    print(header)
    print(f"{'=' * 60}")

    driver.setCruisingSpeed(0)
    driver.setSteeringAngle(0)
    current_speed = 0
    current_steering = 0
    check_time = 0
    check_slip = False

    for _ in range(int(RESET_DELAY * 20)):
        supervisor.step()

    supervisor.simulationResetPhysics()
    for _ in range(5):
        supervisor.step()

    if car_node:
        try:
            rot_field = car_node.getField("rotation")
            if rot_field:
                rot_field.setSFRotation(RESET_ROTATION)
            for _ in range(3):
                supervisor.step()

            trans_field = car_node.getField("translation")
            if trans_field:
                trans_field.setSFVec3f(RESET_POSITION)
            for _ in range(10):
                supervisor.step()
        except Exception as e:
            print(f"⚠️ Ошибка сброса: {e}")

    for _ in range(20):
        driver.setCruisingSpeed(0)
        driver.setSteeringAngle(0)
        supervisor.step()

    in_reset = False
    collision_detected = False
    goal_reached = False
    print(f"🚀 Готов к движению!\n")


# =============================================================================
# ПЕРЕМЕННЫЕ СОСТОЯНИЯ
# =============================================================================
current_speed = 0.0
current_steering = 0.0
collision_detected = False
goal_reached = False
in_reset = False
collision_count = 0
goal_count = 0
last_debug_time = 0

# =============================================================================
# ГЛАВНЫЙ ЦИКЛ
# =============================================================================
print(f"\n{'=' * 60}")
print(f"🎮 УПРАВЛЕНИЕ:")
print(f"   ↑/↓    : Газ/Тормоз")
print(f"   ←/→    : Поворот")
print(f"   Пробел : Экстренная остановка")
print(f"   R      : Ручной сброс")
print(f"   C      : Перегенерация препятствий")
print(f"\n❄️ ФИЗИКА СНЕГА (как в коде друга):")
print(f"   - Замедление зависит от расстояния до центра")
print(f"   - Формирование колеи (проторенный путь)")
print(f"   - Эффект заноса при резких поворотах (коэф. {KOEF_SNOW_SLIP})")
print(f"\n🧊 ФИЗИКА ЛЬДА:")
print(f"   - Базовое замедление: x{ICE_BASE_SPEED}")
print(f"   - Сильное скольжение при поворотах: x{ICE_SLIP_FACTOR}")
print(f"   - Доп. скольжение при резких поворотах: x{ICE_TURN_SLIP}")
print(f"   - НЕТ КОЛЕИ (лёд не сохраняет следы)")
print(f"\n📍 НАЧАЛЬНАЯ ПОЗИЦИЯ: X=0.16, Y=-45, Z=0")
print(f"{'=' * 60}\n")

while supervisor.step() != -1:

    if not in_reset:
        is_goal, dist_goal = check_goal_3d()
        is_collision, collision_info, dist_obs = check_collisions_lidar()

        # Отладочный вывод
        current_time = supervisor.getTime()
        if current_time - last_debug_time > 1.0:
            car_pos = get_car_position()

            # Проверка типа местности
            terrain = "🟫 ДОРОГА"
            in_snow = False
            for i in range(len(snow_fields)):
                dist, max_dist, idx = is_in_snow(car_pos, i)
                if dist is not None:
                    track_val = 0
                    snow_pos = snow_positions[i]
                    local_x = int((car_pos[0] - snow_pos[0] + TRACK_SIZE / 2))
                    local_y = int((car_pos[1] - snow_pos[1] + TRACK_SIZE / 2))
                    if 0 <= local_x < TRACK_SIZE and 0 <= local_y < TRACK_SIZE:
                        track_val = track_maps[i][local_x][local_y]
                    terrain = f"❄️ СНЕГ (дист={dist:.1f}/{max_dist:.1f}, колея={track_val:.2f})"
                    in_snow = True
                    break

            if not in_snow and is_in_ice(car_pos):
                terrain = "🧊 ЛЁД (СКОЛЬЗКО!)"

            print(f"   📊 Преп:{dist_obs:.1f}m | Цель:{dist_goal:.1f}m | {terrain}")
            last_debug_time = current_time

        # Обработка цели
        if is_goal and not goal_reached:
            goal_reached = True
            goal_count += 1
            in_reset = True

            car_pos = get_car_position()
            target_p = target_node.getPosition() if target_node else [0, 0, 0]

            print(f"\n{'🎯' * 30}")
            print(f"🎯 GOAL #{goal_count} ДОСТИГНУТА!")
            print(f"   Машина: X={car_pos[0]:.2f}, Y={car_pos[1]:.2f}")
            print(f"   Цель: X={target_p[0]:.2f}, Y={target_p[1]:.2f}")
            print(f"{'🎯' * 30}")

            driver.setCruisingSpeed(0)
            driver.setSteeringAngle(0)
            current_speed = 0
            reset_car("GOAL")
            continue

        # Обработка столкновения
        elif is_collision and not collision_detected and not is_goal:
            collision_detected = True
            collision_count += 1
            in_reset = True

            car_pos = get_car_position()

            print(f"\n{'💥' * 30}")
            print(f"💥 СТОЛКНОВЕНИЕ #{collision_count}!")
            print(f"   {collision_info}")
            print(f"   Машина: X={car_pos[0]:.2f}, Y={car_pos[1]:.2f}")
            print(f"{'💥' * 30}")

            driver.setCruisingSpeed(0)
            driver.setSteeringAngle(0)
            current_speed = 0

            randomize_obstacles()
            reset_car("COLLISION")
            continue

    # =========================================================================
    # ОБРАБОТКА КЛАВИАТУРЫ
    # =========================================================================
    key = keyboard.getKey()
    while key != -1:
        if key == Keyboard.UP:
            current_speed += 2.0
            if current_speed > MAX_SPEED:
                current_speed = MAX_SPEED
        elif key == Keyboard.DOWN:
            current_speed -= 2.0
            if current_speed < -MAX_SPEED:
                current_speed = -MAX_SPEED
        elif key == Keyboard.RIGHT:
            current_steering += 0.05
            if current_steering > MAX_STEERING_ANGLE:
                current_steering = MAX_STEERING_ANGLE
        elif key == Keyboard.LEFT:
            current_steering -= 0.05
            if current_steering < -MAX_STEERING_ANGLE:
                current_steering = -MAX_STEERING_ANGLE
        elif key == ord(' '):
            current_speed = 0.0
            current_steering = 0.0
            print("🛑 Экстренная остановка!")
        elif key == ord('R') or key == ord('r'):
            in_reset = True
            reset_car("MANUAL")
            continue
        elif key == ord('C') or key == ord('c'):
            randomize_obstacles()
        key = keyboard.getKey()

    # =========================================================================
    # ПРИМЕНЕНИЕ ФИЗИКИ (как в коде друга)
    # =========================================================================
    car_pos = get_car_position()

    # Обновляем карту колеи (только для снега)
    update_track(car_pos, current_speed, current_steering)

    # Применяем физику снега
    final_speed = apply_snow_physics(car_pos, current_speed, current_steering)

    # Применяем физику льда (отдельно, перекрывает снег если на льду)
    if is_in_ice(car_pos):
        final_speed = apply_ice_physics(car_pos, current_speed, current_steering)

    # Отправка команд
    if not collision_detected and not goal_reached:
        driver.setCruisingSpeed(final_speed)
        driver.setSteeringAngle(current_steering)