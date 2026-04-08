"""keyboard_control controller with snow support."""

from vehicle import Driver
from controller import Keyboard, Supervisor
import math

# --- Инициализация ---
driver = Driver()
supervisor = Supervisor()
keyboard = Keyboard()
keyboard.enable(50)

# --- Константы ---
MAX_SPEED = 50.0
MAX_STEERING_ANGLE = 0.5
SNOW_SPEED_FACTOR = 0.3  # В снегу скорость 30%

# --- Получаем доступ к ноде автомобиля ---
car_node = supervisor.getFromDef("CAR")
if car_node is None:
    print("ОШИБКА: Автомобиль с DEF 'CAR' не найден!")

# --- Получаем все снежные поля (поиск по типу SnowField и по имени) ---
snow_fields = []
root = supervisor.getRoot()
children = root.getField("children")

# Сначала ищем по имени "snow_field" в любых нодах
for i in range(children.getCount()):
    node = children.getMFNode(i)
    # Проверяем имя ноды
    try:
        name_field = node.getField("name")
        if name_field:
            name = name_field.getSFString()
            if "snow" in name.lower():  # Ищем любые ноды с snow в имени
                pos = node.getField("translation").getSFVec3f()
                snow_fields.append(node)
                print(f"❄️ Найдено снежное поле: '{name}' на позиции ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
    except:
        # Если нет поля name, пропускаем
        pass

# Также ищем Solid ноды, которые могут быть снежными полями
for i in range(children.getCount()):
    node = children.getMFNode(i)
    if node.getTypeName() == "Solid" or node.getTypeName() == "SnowField":
        try:
            name_field = node.getField("name")
            if name_field:
                name = name_field.getSFString()
                if "snow" in name.lower() and node not in snow_fields:
                    pos = node.getField("translation").getSFVec3f()
                    snow_fields.append(node)
                    print(f"❄️ Найдено Solid снежное поле: '{name}' на позиции ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        except:
            pass

print(f"\n🚗 Управление автомобилем:")
print(f"  ↑/↓ : Газ/Тормоз")
print(f"  ←/→ : Поворот")
print(f"  Пробел : Экстренная остановка")
print(f"Найдено снежных полей: {len(snow_fields)}")
print(f"(Кликните по 3D виду для активации управления!)")

# --- Переменные состояния ---
current_speed = 0.0
current_steering = 0.0
last_snow_message_time = 0


# --- Функция проверки нахождения в снегу (исправленная) ---
def is_in_any_snow(car_pos, snow_fields):
    """Проверяет, находится ли машина в каком-либо снежном поле"""
    if not snow_fields:
        return False

    for snow in snow_fields:
        try:
            snow_pos = snow.getField("translation").getSFVec3f()

            # Получаем размер снежного поля
            size_field = snow.getField("size")
            if size_field:
                snow_size = size_field.getSFFloat()
            else:
                # Если используете sizeX, sizeY, sizeZ
                sizeX_field = snow.getField("sizeX")
                if sizeX_field:
                    snow_size = sizeX_field.getSFFloat()
                else:
                    snow_size = 4.0  # Размер по умолчанию

            # Получаем высоту снежного поля (если есть sizeY)
            sizeY_field = snow.getField("sizeY")
            if sizeY_field:
                snow_height = sizeY_field.getSFFloat()
            else:
                snow_height = snow_size  # Если нет sizeY, используем size

            # Проверяем по X и Z координатам
            in_x = abs(car_pos[0] - snow_pos[0]) < snow_size / 2
            in_z = abs(car_pos[2] - snow_pos[2]) < snow_size / 2

            # Проверяем по Y (высота) - машина должна быть на уровне снега
            # Нижняя часть машины примерно на высоте колёс (чуть выше центра)
            car_bottom_y = car_pos[1] - 0.8  # Приблизительная высота до низа машины
            snow_bottom_y = snow_pos[1] - snow_height / 2
            snow_top_y = snow_pos[1] + snow_height / 2

            # Машина касается снега, если её нижняя часть в пределах высоты снега
            in_y = snow_bottom_y <= car_bottom_y <= snow_top_y

            # Машина в снегу только если она внутри по всем трём осям
            if in_x and in_y and in_z:
                return True
        except Exception as e:
            # Если не можем получить поля, пропускаем эту ноду
            continue

    return False

# --- Главный цикл ---
while supervisor.step() != -1:
    # Обработка клавиш
    key = keyboard.getKey()
    while key != -1:
        if key == Keyboard.UP:
            current_speed += 2.0
            if current_speed > MAX_SPEED:
                current_speed = MAX_SPEED
            print(f"Скорость: {current_speed:.1f} км/ч")
        elif key == Keyboard.DOWN:
            current_speed -= 2.0
            if current_speed < -MAX_SPEED:
                current_speed = -MAX_SPEED
            print(f"Скорость: {current_speed:.1f} км/ч")
        elif key == Keyboard.RIGHT:
            current_steering += 0.05
            if current_steering > MAX_STEERING_ANGLE:
                current_steering = MAX_STEERING_ANGLE
            print(f"Поворот: {current_steering:.2f} рад")
        elif key == Keyboard.LEFT:
            current_steering -= 0.05
            if current_steering < -MAX_STEERING_ANGLE:
                current_steering = -MAX_STEERING_ANGLE
            print(f"Поворот: {current_steering:.2f} рад")
        elif key == ord(' '):
            current_speed = 0.0
            current_steering = 0.0
            print("🛑 Экстренная остановка!")
        key = keyboard.getKey()

    # --- Проверка на снег ---
    final_speed = current_speed

    if car_node and snow_fields:
        car_pos = car_node.getPosition()

        if is_in_any_snow(car_pos, snow_fields):  # Только 2 параметра!
            final_speed = current_speed * SNOW_SPEED_FACTOR
            # Визуальная индикация (не чаще чем раз в секунду)
            current_time = supervisor.getTime()
            if current_time - last_snow_message_time > 1.0:
                print("❄️ В снегу! Скорость снижена ❄️")
                last_snow_message_time = current_time

    # Отправляем команды автомобилю
    driver.setCruisingSpeed(final_speed)
    driver.setSteeringAngle(current_steering)