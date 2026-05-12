import re
import json


def simple_parse_wbt(wbt_path):
    """Простой парсинг WBT файла"""

    with open(wbt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Поиск всех Solid объектов с translation
    pattern = r'translation\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)[^}]*name\s+"([^"]+)"'

    obstacles = []
    ice_zones = []
    snow_zones = []

    for match in re.finditer(pattern, content):
        x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
        name = match.group(4)

        obj = [x, y, 2.0, 2.0]  # Стандартный размер 2x2

        if 'ice' in name.lower():
            ice_zones.append(obj)
        elif 'snow' in name.lower():
            snow_zones.append(obj)
        elif 'target' in name.lower() or 'finish' in name.lower():
            pass  # Игнорируем цели
        else:
            obstacles.append(obj)

    world_data = {
        "obstacles": obstacles,
        "ice_zones": ice_zones,
        "snow_zones": snow_zones
    }

    with open("world_data.json", 'w') as f:
        json.dump(world_data, f, indent=2)

    print(f"Найдено: {len(obstacles)} препятствий, {len(ice_zones)} льда, {len(snow_zones)} снега")
    return world_data


if __name__ == "__main__":
    simple_parse_wbt("world.wbt")