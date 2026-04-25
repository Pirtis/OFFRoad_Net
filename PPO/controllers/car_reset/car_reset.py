from controller import Supervisor
import socket
import random
import math
import json
import os

supervisor = Supervisor()

# Настройка UDP-сокета
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 6006))
sock.setblocking(False)

waypoint_cubes = []


def distance_2d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def random_position(z_check=False):
    x = random.uniform(-90, 200)
    y = random.uniform(-170, 110)
    if z_check:
        z = random.uniform(-0.9, 0)
    else:
        z = 0
    return [x, y, z]


def valid_object_position(pos, car_pos, finish_pos, min_dist=10):
    if distance_2d(pos, car_pos) < min_dist:
        return False
    if distance_2d(pos, finish_pos) < min_dist:
        return False
    return True


def is_point_in_obstacle(x, y, obstacles):
    for obs in obstacles:
        ox, oy, w, d = obs
        if (abs(x - ox) < w/2 and abs(y - oy) < d/2):
            return True
    return False


def create_waypoint_cube(x, y, z, index):
    root = supervisor.getRoot()
    children_field = root.getField("children")

    cube_string = """
    Solid {
      translation %f %f %f
      children [
        Shape {
          appearance PBRAppearance {
            baseColor 0 0 1
            metalness 0
            roughness 0.5
            transparency 0.3
          }
          geometry Box {
            size 1 1 1
          }
        }
      ]
      name "waypoint_%d"
      boundingObject Box {
        size 1 1 1
      }
      contactMaterial "default"
    }
    """ % (x, y, z, index)

    try:
        try:
            cube = children_field.importMFNodeFromString(-1, cube_string)
        except:
            cube = children_field.importMFNode(-1, cube_string)
        return cube
    except Exception as e:
        print(f"Failed to create waypoint cube {index} at ({x}, {y}): {e}")
        return None


def clear_waypoint_cubes():
    global waypoint_cubes

    root = supervisor.getRoot()
    children_field = root.getField("children")

    i = 0
    while i < children_field.getCount():
        node = children_field.getMFNode(i)
        try:
            name_field = node.getField("name")
            if name_field:
                name = name_field.getSFString()
                if name.startswith("waypoint_"):
                    children_field.removeMF(i)
                    continue
        except:
            pass
        i += 1

    waypoint_cubes = []
    print("Waypoint cubes cleared")


def spawn_waypoint_cubes(waypoints):
    global waypoint_cubes

    obstacles = []
    try:
        possible_paths = [
            "world_data.json",
            "../ppo_waypoint_controller/world_data.json",
            "../../controllers/ppo_waypoint_controller/world_data.json"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    obstacles = data.get("obstacles", [])
                break
    except:
        pass

    clear_waypoint_cubes()

    root = supervisor.getRoot()
    children_field = root.getField("children")

    valid_waypoints = []
    for i, wp in enumerate(waypoints):
        if is_point_in_obstacle(wp[0], wp[1], obstacles):
            print(f"Waypoint {i} is inside obstacle, skipping: ({wp[0]:.1f}, {wp[1]:.1f})")
            continue
        valid_waypoints.append(wp)

    if len(valid_waypoints) < 2 and len(waypoints) >= 2:
        print("All waypoints inside obstacles, using first and last points")
        valid_waypoints = [waypoints[0], waypoints[-1]]

    for i, wp in enumerate(valid_waypoints):
        try:
            cube_string = """
            Solid {
              translation %f %f 0.5
              children [
                Shape {
                  appearance PBRAppearance {
                    baseColor 0 0 1
                    metalness 0
                    roughness 0.5
                    transparency 0.3
                  }
                  geometry Box {
                    size 1 1 1
                  }
                }
              ]
              name "waypoint_%d"
              boundingObject Box {
                size 1 1 1
              }
              contactMaterial "default"
            }
            """ % (wp[0], wp[1], i)

            try:
                children_field.importMFNodeFromString(-1, cube_string)
            except:
                children_field.importMFNode(-1, cube_string)

            waypoint_cubes.append(i)
            print(f"Created waypoint cube {i} at ({wp[0]:.1f}, {wp[1]:.1f})")
        except Exception as e:
            print(f"Error creating waypoint cube {i}: {e}")

    print(f"Successfully created {len(valid_waypoints)} waypoint cubes")


def load_world_data():
    obstacles = []
    ice_zones = []
    snow_zones = []

    try:
        possible_paths = [
            "world_data.json",
            "../ppo_waypoint_controller/world_data.json",
            "../../controllers/ppo_waypoint_controller/world_data.json"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    obstacles = data.get("obstacles", [])
                    ice_zones = data.get("ice_zones", [])
                    snow_zones = data.get("snow_zones", [])
                print(f"Loaded world data from {path}")
                print(f"   Obstacles: {len(obstacles)}")
                print(f"   Ice zones: {len(ice_zones)}")
                print(f"   Snow zones: {len(snow_zones)}")
                break
        else:
            print("world_data.json not found, using default configuration")
    except Exception as e:
        print(f"Error loading world_data.json: {e}")

    return obstacles, ice_zones, snow_zones


def respawn_vehicle(car_node, finish_node, objects_to_move, snow_to_move):
    print("\nStarting world regeneration...")

    attempt = 0
    while True:
        car_pos = [random.uniform(-90, 200), random.uniform(-170, 110), 0.969985]
        finish_pos = [random.uniform(-90, 200), random.uniform(-170, 110), 0.01]

        if distance_2d(car_pos, finish_pos) >= 100:
            with open("finish_pos.txt", "w") as f:
                f.write(f"{finish_pos[0]:.1f} {finish_pos[1]:.1f} {finish_pos[2]}")
            print(f"Finish positioned at: ({finish_pos[0]:.1f}, {finish_pos[1]:.1f})")
            break

        attempt += 1
        if attempt > 100:
            finish_pos = [random.uniform(-90, 200), random.uniform(-170, 110), 0.01]
            with open("finish_pos.txt", "w") as f:
                f.write(f"{finish_pos[0]:.1f} {finish_pos[1]:.1f} {finish_pos[2]}")
            print(f"Finish positioned (forced): ({finish_pos[0]:.1f}, {finish_pos[1]:.1f})")
            break

    start_rotation = [0.012688, 0.00732535, -0.999893, -1.57093]
    car_node.getField("translation").setSFVec3f(car_pos)
    car_node.getField("rotation").setSFRotation(start_rotation)
    car_node.resetPhysics()
    print(f"Vehicle positioned at: ({car_pos[0]:.1f}, {car_pos[1]:.1f})")

    finish_node.getField("translation").setSFVec3f(finish_pos)
    finish_node.resetPhysics()

    moved_objects = 0
    for obj in objects_to_move:
        for attempt in range(30):
            pos = random_position(False)

            if valid_object_position(pos, car_pos, finish_pos, min_dist=8):
                valid = True
                for other in objects_to_move:
                    if other == obj:
                        continue
                    try:
                        other_pos = other.getField("translation").getSFVec3f()
                        if distance_2d(pos, other_pos) < 5:
                            valid = False
                            break
                    except:
                        pass

                if valid:
                    obj.getField("translation").setSFVec3f(pos)
                    obj.resetPhysics()
                    moved_objects += 1
                    break

    print(f"Objects relocated: {moved_objects}")

    snow_positions = []
    moved_snow = 0

    for obj in snow_to_move:
        for attempt in range(20):
            pos = random_position(True)

            if valid_object_position(pos, car_pos, finish_pos, min_dist=10):
                snow_positions.append(pos)
                obj.getField("translation").setSFVec3f(pos)
                obj.resetPhysics()
                moved_snow += 1
                break

    with open("snow.txt", "w") as f:
        for p in snow_positions:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

    print(f"Snow zones relocated: {moved_snow}")
    print("World regeneration completed\n")

    return car_pos, finish_pos


def find_all_objects(root_children):
    car_node = None
    finish_node = None
    objects_to_move = []
    snow_to_move = []

    obstacles_data, ice_zones_data, snow_zones_data = load_world_data()

    for i in range(root_children.getCount()):
        node = root_children.getMFNode(i)
        try:
            name_field = node.getField("name")
            if name_field is None:
                continue

            name = name_field.getSFString()

            if name == "vehicle":
                car_node = node
            elif name == "finish":
                finish_node = node
            elif "TARGET" in name or "target" in name.lower() or "Building" in name:
                objects_to_move.append(node)
            elif "ice" in name.lower():
                objects_to_move.append(node)
            elif "snow" in name.lower():
                snow_to_move.append(node)
        except:
            continue

    print(f"Objects found:")
    print(f"   - Obstacles and ice: {len(objects_to_move)}")
    print(f"   - Snow zones: {len(snow_to_move)}")

    return car_node, finish_node, objects_to_move, snow_to_move


print("=" * 60)
print("CAR_RESET - Supervisor for PPO Waypoint Navigator")
print("=" * 60)
print("Supported commands:")
print("   - RESET                 : Full world regeneration")
print("   - WAYPOINTS:x,y;x,y    : Spawn waypoint cubes")
print("   - CLEAR_WAYPOINTS      : Remove all waypoint cubes")
print("=" * 60)
print("Waiting for signals...\n")

while supervisor.step() != -1:
    try:
        data, addr = sock.recvfrom(1024)
        message = data.decode()

        if message == "RESET":
            print("\n" + "=" * 60)
            print("RESET signal received")
            print("=" * 60)

            clear_waypoint_cubes()

            root_children = supervisor.getRoot().getField("children")
            car_node, finish_node, objects_to_move, snow_to_move = find_all_objects(root_children)

            if car_node is None:
                print("Error: Vehicle node not found")
                continue

            if finish_node is None:
                print("Error: Finish node not found")
                continue

            car_pos, finish_pos = respawn_vehicle(car_node, finish_node, objects_to_move, snow_to_move)

            print(f"Distance to finish: {distance_2d(car_pos, finish_pos):.1f}")
            print("=" * 60 + "\n")

        elif message.startswith("WAYPOINTS:"):
            waypoints_str = message[10:]
            waypoints = []

            for point in waypoints_str.split(";"):
                if point:
                    coords = point.split(",")
                    if len(coords) == 2:
                        try:
                            waypoints.append((float(coords[0]), float(coords[1])))
                        except:
                            pass

            if waypoints:
                spawn_waypoint_cubes(waypoints)
                print(f"Spawned {len(waypoints)} waypoint cubes")
            else:
                print("Empty waypoints received")

        elif message == "CLEAR_WAYPOINTS":
            clear_waypoint_cubes()
            print("Waypoint cubes cleared by request")

        elif message == "STATUS":
            print("Supervisor is running")

    except socket.error:
        pass
    except Exception as e:
        print(f"Error: {e}")