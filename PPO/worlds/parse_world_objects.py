import json
import os
import re


def parse_with_line_by_line(wbt_path):
    """Построчный парсинг с улучшенным поиском препятствий"""

    with open(wbt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    obstacles = []
    ice_zones = []
    snow_zones = []
    finish_pos = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Поиск начала объекта
        if 'Solid {' in line or ('DEF' in line and ('Solid' in line or 'Building' in line)):
            # Собираем весь объект
            obj_lines = [line]
            brace_count = line.count('{') - line.count('}')
            i += 1

            while i < len(lines) and brace_count > 0:
                obj_lines.append(lines[i])
                brace_count += lines[i].count('{') - lines[i].count('}')
                i += 1

            obj_text = ' '.join(obj_lines).lower()

            # 1. ПОИСК СНЕГА
            if 'elevationgrid' in obj_text and ('snow' in obj_text):
                coords = extract_coordinates(obj_lines)
                if coords:
                    snow_zones.append([coords[0], coords[1], 100.0, 100.0])
                    print(f"❄️ Снег: ({coords[0]:.1f}, {coords[1]:.1f})")

            # 2. ПОИСК ЛЬДА
            elif 'contactmaterial "ice"' in obj_text:
                coords = extract_coordinates(obj_lines)
                size = extract_size_from_plane(obj_lines)
                if coords:
                    ice_zones.append([coords[0], coords[1], size[0], size[1]])
                    print(f"🧊 Лед: ({coords[0]:.1f}, {coords[1]:.1f}) размер {size[0]:.1f}x{size[1]:.1f}")

            # 3. ПОИСК ФИНИША
            elif 'name "finish"' in obj_text:
                coords = extract_coordinates(obj_lines)
                if coords:
                    finish_pos = coords[:2]
                    print(f"🏁 Финиш: ({coords[0]:.1f}, {coords[1]:.1f})")

            # 4. ПОИСК ПРЕПЯТСТВИЙ (TARGET, Building)
            elif ('target' in obj_text or 'building' in obj_text) and 'elevationgrid' not in obj_text:
                # Проверяем что это не лед и не снег
                if 'ice' not in obj_text and 'snow' not in obj_text:
                    coords = extract_coordinates(obj_lines)

                    # Определяем размер
                    width, depth = 4.0, 4.0

                    # Ищем corners (для Building)
                    corners = extract_corners(obj_lines)
                    if corners:
                        width = abs(corners[0] - corners[2])
                        depth = abs(corners[1] - corners[3])
                    else:
                        # Ищем size (для SolidBox)
                        size = extract_size_from_box(obj_lines)
                        if size:
                            width, depth = size[0], size[1]

                    if coords:
                        obstacles.append([coords[0], coords[1], width, depth])
                        print(f"🏢 Препятствие: ({coords[0]:.1f}, {coords[1]:.1f}) размер {width:.1f}x{depth:.1f}")

            # 5. Дополнительный поиск для DEF TARGET
            elif 'def target' in obj_text or 'name "target' in obj_text:
                coords = extract_coordinates(obj_lines)
                if coords and 'ice' not in obj_text and 'snow' not in obj_text:
                    obstacles.append([coords[0], coords[1], 4.0, 4.0])
                    print(f"🎯 TARGET: ({coords[0]:.1f}, {coords[1]:.1f})")
        else:
            i += 1

    return {
        "obstacles": obstacles,
        "ice_zones": ice_zones,
        "snow_zones": snow_zones,
        "finish_position": finish_pos
    }


def extract_coordinates(lines):
    """Извлечение координат из строк объекта"""
    for line in lines:
        if 'translation' in line.lower():
            # Ищем числа в строке
            numbers = []
            parts = line.split()
            for part in parts:
                try:
                    num = float(part)
                    numbers.append(num)
                except:
                    pass

            if len(numbers) >= 3:
                return numbers[:3]
    return None


def extract_size_from_plane(lines):
    """Извлечение размера из Plane"""
    for line in lines:
        if 'plane' in line.lower():
            for next_line in lines:
                if 'size' in next_line.lower():
                    numbers = []
                    parts = next_line.split()
                    for part in parts:
                        try:
                            num = float(part)
                            numbers.append(num)
                        except:
                            pass
                    if len(numbers) >= 2:
                        return [numbers[0], numbers[1]]
    return [10.0, 10.0]


def extract_size_from_box(lines):
    """Извлечение размера из SolidBox"""
    for line in lines:
        if 'box' in line.lower():
            for next_line in lines:
                if 'size' in next_line.lower():
                    numbers = []
                    parts = next_line.split()
                    for part in parts:
                        try:
                            num = float(part)
                            numbers.append(num)
                        except:
                            pass
                    if len(numbers) >= 3:
                        return [numbers[0], numbers[1]]
    return None


def extract_corners(lines):
    """Извлечение углов из Building"""
    for line in lines:
        if 'corners' in line.lower():
            numbers = []
            parts = line.split()
            for part in parts:
                try:
                    num = float(part)
                    numbers.append(num)
                except:
                    pass

            if len(numbers) >= 4:
                return numbers[:4]
    return None


def search_all_objects_direct(wbt_path):
    """Прямой поиск всех объектов по ключевым словам"""

    with open(wbt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    obstacles = []
    ice_zones = []
    snow_zones = []

    # Разделяем на блоки по DEF
    lines = content.split('\n')
    current_block = []
    in_block = False

    for line in lines:
        if line.strip().startswith('DEF ') or line.strip().startswith('Solid {') or line.strip().startswith(
                'Building {'):
            if current_block:
                process_block(current_block, obstacles, ice_zones, snow_zones)
                current_block = []
            current_block.append(line)
            in_block = True
        elif in_block:
            current_block.append(line)
            if line.strip() == '}':
                process_block(current_block, obstacles, ice_zones, snow_zones)
                current_block = []
                in_block = False

    return obstacles, ice_zones, snow_zones


def process_block(block_lines, obstacles, ice_zones, snow_zones):
    """Обработка одного блока"""
    block_text = ' '.join(block_lines).lower()

    # Поиск координат
    coords = None
    for line in block_lines:
        if 'translation' in line.lower():
            numbers = []
            parts = line.split()
            for part in parts:
                try:
                    numbers.append(float(part))
                except:
                    pass
            if len(numbers) >= 3:
                coords = numbers[:3]
                break

    if not coords:
        return

    # Определяем тип объекта
    # 1. Лед
    if 'contactmaterial "ice"' in block_text:
        ice_zones.append([coords[0], coords[1], 50.0, 20.0])
        print(f"🧊 Лед: ({coords[0]:.1f}, {coords[1]:.1f})")

    # 2. Снег
    elif 'elevationgrid' in block_text and ('snow' in block_text):
        snow_zones.append([coords[0], coords[1], 100.0, 100.0])
        print(f"❄️ Снег: ({coords[0]:.1f}, {coords[1]:.1f})")

    # 3. Препятствия (TARGET, Building)
    elif ('target' in block_text or 'building' in block_text) and 'ice' not in block_text:
        # Определяем размер
        width, depth = 4.0, 4.0

        # Ищем corners
        for line in block_lines:
            if 'corners' in line.lower():
                numbers = []
                parts = line.split()
                for part in parts:
                    try:
                        numbers.append(float(part))
                    except:
                        pass
                if len(numbers) >= 4:
                    width = abs(numbers[0] - numbers[2])
                    depth = abs(numbers[1] - numbers[3])
                    break

        obstacles.append([coords[0], coords[1], width, depth])
        print(f"🏢 Препятствие: ({coords[0]:.1f}, {coords[1]:.1f}) размер {width:.1f}x{depth:.1f}")


def create_world_data(wbt_path, output_path="world_data.json"):
    """Создание файла world_data.json"""

    print(f"\n📁 Парсинг файла: {wbt_path}")
    print("-" * 60)

    # Используем прямой поиск
    obstacles, ice_zones, snow_zones = search_all_objects_direct(wbt_path)

    # Также пробуем построчный парсинг
    data = parse_with_line_by_line(wbt_path)

    # Объединяем результаты
    all_obstacles = obstacles + data["obstacles"]
    # Удаляем дубликаты
    unique_obstacles = []
    for obs in all_obstacles:
        if obs not in unique_obstacles:
            unique_obstacles.append(obs)

    simple_data = {
        "obstacles": unique_obstacles,
        "ice_zones": ice_zones + data["ice_zones"],
        "snow_zones": snow_zones + data["snow_zones"]
    }

    # Сохраняем JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(simple_data, f, indent=2, ensure_ascii=False)

    # Сохраняем финиш
    if data["finish_position"]:
        with open("finish_pos.txt", "w") as f:
            f.write(f"{int(data['finish_position'][0])} {int(data['finish_position'][1])} 0")
        print(f"\n🎯 Финиш: ({data['finish_position'][0]:.1f}, {data['finish_position'][1]:.1f})")

    print("-" * 60)
    print(f"✅ Файл сохранен: {output_path}")
    print(f"\n📊 Статистика:")
    print(f"   - Препятствий: {len(simple_data['obstacles'])}")
    print(f"   - Ледяных зон: {len(simple_data['ice_zones'])}")
    print(f"   - Снежных зон: {len(simple_data['snow_zones'])}")

    return simple_data


if __name__ == "__main__":
    wbt_file = "moose_demo.wbt"

    if not os.path.exists(wbt_file):
        print(f"❌ Файл {wbt_file} не найден!")
        exit(1)

    world_data = create_world_data(wbt_file)

    # Если препятствий все еще 0, добавим вручную
    if len(world_data['obstacles']) == 0:
        print("\n⚠️ Препятствия не найдены. Добавьте их вручную (y/n)?")
        answer = input().strip().lower()
        if answer == 'y':
            print("\nВведите препятствия в формате: x y width depth")
            print("Пример: 185.983 76.4116 4 4")
            print("Пустая строка для завершения:")

            while True:
                line = input().strip()
                if not line:
                    break
                try:
                    parts = list(map(float, line.split()))
                    if len(parts) == 4:
                        world_data['obstacles'].append(parts)
                        print(f"  ✅ Добавлено: {parts}")
                    else:
                        print("  ❌ Нужно 4 числа")
                except:
                    print("  ❌ Ошибка ввода")

            # Сохраняем
            with open("world_data.json", "w") as f:
                json.dump(world_data, f, indent=2)
            print(f"\n✅ Сохранено {len(world_data['obstacles'])} препятствий")