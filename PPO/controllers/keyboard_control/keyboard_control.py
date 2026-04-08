#!/usr/bin/env python3
"""Keyboard controller with ghost target (no physics, visible but lidar-transparent)."""

from vehicle import Driver
from controller import Keyboard, Supervisor, Lidar, GPS
import math

# --- Инициализация ---
driver = Driver()
supervisor = Supervisor()
keyboard = Keyboard()
keyboard.enable(50)

# --- Константы ---
MAX_SPEED = 50.0
MAX_STEERING_ANGLE = 0.5
SNOW_SPEED_FACTOR = 0.3

# Параметры столкновений
COLLISION_DISTANCE = 0.5  # Для всех препятствий включая стены
GOAL_DISTANCE = 4.0  # Для цели (зелёный куб) - только 3D!
RESET_DELAY = 2.0
RESET_POSITION = [-11.5, 0.5, 0]
RESET_ROTATION = [0, 1, 0, 0]

# --- Получаем доступ к нодам ---
car_node = supervisor.getFromDef("CAR")
if car_node is None:
    print("⚠️ CAR не найден, используем getSelf()")
    car_node = driver.getSelf()

obstacle_node = supervisor.getFromDef("OBSTACLE")
if obstacle_node:
    print(f"✅ Препятствие (OBSTACLE) найдено")
else:
    print("⚠️ Препятствие не найдено")

target_node = supervisor.getFromDef("TARGET")
target_pos = None

if target_node:
    target_pos = target_node.getPosition()
    print(f"✅ Цель (TARGET) найдена на ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")

    # !!! ОТКЛЮЧАЕМ ФИЗИКУ ЦЕЛИ !!!
    try:
        # Убираем boundingObject (физическую форму)
        target_node.removeBoundingObject()
        print("   👻 Физика цели отключена (призрачный режим)")

        # Отключаем коллизии полностью
        target_node.enableContactPointsTracking(False)

        # Делаем ноду статичной без физики
        target_node.resetPhysics()

    except Exception as e:
        print(f"   ⚠️ Не удалось отключить физику: {e}")
else:
    print("❌ Цель не найдена!")

# --- Инициализация ЛИДАРОВ ---
lidar_front = driver.getDevice("lidar_front")
lidar_left = driver.getDevice("lidar_left")
lidar_right = driver.getDevice("lidar_right")

for lidar, name in [(lidar_front, "front"), (lidar_left, "left"), (lidar_right, "right")]:
    if lidar:
        lidar.enable(50)
        fov = lidar.getFov()
        res = lidar.getHorizontalResolution()
        max_range = lidar.getMaxRange()
        print(f"✅ Лидар {name}: FOV={math.degrees(fov):.1f}°, res={res}, range={max_range}m")

# --- Инициализация GPS ---
gps = driver.getDevice("gps")
if gps:
    gps.enable(50)
    print(f"✅ GPS активирован")

# --- Поиск снежных полей ---
snow_fields = []
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
                print(f"❄️ Снежное поле: '{name}' ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
    except:
        pass

print(f"\n{'=' * 60}")
print(f"🎮 УПРАВЛЕНИЕ:")
print(f"   ↑/↓    : Газ/Тормоз")
print(f"   ←/→    : Поворот")
print(f"   Пробел : Экстренная остановка")
print(f"   R      : Ручной сброс позиции")
print(f"\n⚠️  Столкновение: лидар < {COLLISION_DISTANCE}m")
print(f"🎯 Цель: 3D-расстояние < {GOAL_DISTANCE}m (призрачная, нет физики)")
print(f"{'=' * 60}")

# --- Переменные состояния ---
current_speed = 0.0
current_steering = 0.0
last_snow_message_time = 0
collision_detected = False
goal_reached = False
in_reset = False
collision_count = 0
goal_count = 0
last_debug_time = 0


def get_car_position():
    """Получение позиции машины."""
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


def get_lidar_min_distance(lidar, name):
    """
    Простое получение минимального расстояния с лидара.
    Цель уже призрачная, ничего не фильтруем!
    """
    if not lidar:
        return float('inf')

    data = lidar.getRangeImage()
    if not data or len(data) == 0:
        return float('inf')

    # Просто минимум валидных значений
    valid = [x for x in data if 0.1 < x < 100]
    return min(valid) if valid else float('inf')


def check_goal_3d():
    """Проверка достижения цели по 3D-расстоянию."""
    if target_node is None:
        return False, float('inf')

    car_pos = get_car_position()
    # Обновляем позицию цели (может двигаться)
    target_pos_current = target_node.getPosition()

    dx = car_pos[0] - target_pos_current[0]
    dy = car_pos[1] - target_pos_current[1]
    dz = car_pos[2] - target_pos_current[2]
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)

    return distance < GOAL_DISTANCE, distance


def check_collisions_lidar():
    """
    Проверка столкновений по лидарам.
    Цель призрачная, поэтому не мешает!
    """
    front_dist = get_lidar_min_distance(lidar_front, "front")
    left_dist = get_lidar_min_distance(lidar_left, "left")
    right_dist = get_lidar_min_distance(lidar_right, "right")

    min_dist = min(front_dist, left_dist, right_dist)

    if front_dist < COLLISION_DISTANCE:
        return True, f"FRONT lidar={front_dist:.2f}m", min_dist
    elif left_dist < COLLISION_DISTANCE:
        return True, f"LEFT lidar={left_dist:.2f}m", min_dist
    elif right_dist < COLLISION_DISTANCE:
        return True, f"RIGHT lidar={right_dist:.2f}m", min_dist

    return False, None, min_dist


def reset_car(reason="RESET"):
    """Сброс машины на исходную позицию."""
    global in_reset, collision_detected, goal_reached, current_speed, current_steering

    if reason == "GOAL":
        header = "🎯 GOAL REACHED! 🎯"
        msg = "Цель достигнута! Возвращаемся на старт..."
    elif reason == "COLLISION":
        header = "💥 COLLISION! 💥"
        msg = "Столкновение! Сброс..."
    else:
        header = "🔄 РУЧНОЙ СБРОС"
        msg = "Возвращаемся на старт..."

    print(f"\n{'=' * 60}")
    print(header)
    print(msg)
    print(f"{'=' * 60}")

    driver.setCruisingSpeed(0)
    driver.setSteeringAngle(0)
    current_speed = 0
    current_steering = 0

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

    if reason == "GOAL":
        print(f"🏆 Готов к следующей попытке!")
    else:
        print(f"🚀 Готов к движению!")
    print(f"{'=' * 60}\n")


def is_in_any_snow(car_pos, snow_fields):
    """Проверка нахождения в снегу."""
    if not snow_fields:
        return False
    for snow in snow_fields:
        try:
            snow_pos = snow.getField("translation").getSFVec3f()
            size_field = snow.getField("size")
            snow_size = size_field.getSFFloat() if size_field else 4.0
            sizeY_field = snow.getField("sizeY")
            snow_height = sizeY_field.getSFFloat() if sizeY_field else snow_size

            in_x = abs(car_pos[0] - snow_pos[0]) < snow_size / 2
            in_z = abs(car_pos[2] - snow_pos[2]) < snow_size / 2
            car_bottom_y = car_pos[1] - 0.8
            snow_bottom_y = snow_pos[1] - snow_height / 2
            snow_top_y = snow_pos[1] + snow_height / 2
            in_y = snow_bottom_y <= car_bottom_y <= snow_top_y

            if in_x and in_y and in_z:
                return True
        except:
            continue
    return False


# --- Главный цикл ---
while supervisor.step() != -1:

    if not in_reset:
        # Проверяем цель (приоритет!)
        is_goal, dist_goal = check_goal_3d()

        # Проверяем столкновения (лидары)
        is_collision, collision_info, dist_obs = check_collisions_lidar()

        # Отладочный вывод
        current_time = supervisor.getTime()
        if current_time - last_debug_time > 1.0:
            obs_str = f"{dist_obs:.1f}" if dist_obs < 100 else "INF"
            goal_str = f"{dist_goal:.1f}" if dist_goal < 100 else "INF"
            print(f"   📊 Obstacle:{obs_str}m | Goal:{goal_str}m")
            last_debug_time = current_time

        # Обработка цели (приоритет!)
        if is_goal and not goal_reached:
            goal_reached = True
            goal_count += 1
            in_reset = True

            car_pos = get_car_position()
            target_p = target_node.getPosition() if target_node else [0, 0, 0]

            print(f"\n{'🎯' * 30}")
            print(f"🎯 GOAL #{goal_count} ДОСТИГНУТА! 🎯")
            print(f"   3D-расстояние: {dist_goal:.3f}m")
            print(f"   Машина:  X={car_pos[0]:.2f} Y={car_pos[1]:.2f} Z={car_pos[2]:.2f}")
            print(f"   Цель:    X={target_p[0]:.2f} Y={target_p[1]:.2f} Z={target_p[2]:.2f}")
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
            print(f"💥 СТОЛКНОВЕНИЕ #{collision_count}! 💥")
            print(f"   {collision_info}")
            print(f"   Машина: X={car_pos[0]:.2f} Y={car_pos[1]:.2f} Z={car_pos[2]:.2f}")
            print(f"{'💥' * 30}")

            driver.setCruisingSpeed(0)
            driver.setSteeringAngle(0)
            current_speed = 0

            reset_car("COLLISION")
            continue

    # --- Обработка клавиатуры ---
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
            print(f"\n⚠️ Ручной сброс")
            in_reset = True
            reset_car("MANUAL")
            continue
        key = keyboard.getKey()

    # --- Проверка на снег ---
    final_speed = current_speed
    if car_node and snow_fields:
        car_pos = get_car_position()
        if is_in_any_snow(car_pos, snow_fields):
            final_speed = current_speed * SNOW_SPEED_FACTOR
            current_time = supervisor.getTime()
            if current_time - last_snow_message_time > 1.0:
                print("❄️ В снегу! Скорость снижена")
                last_snow_message_time = current_time

    # --- Отправка команд ---
    if not collision_detected and not goal_reached:
        driver.setCruisingSpeed(final_speed)
        driver.setSteeringAngle(current_steering)