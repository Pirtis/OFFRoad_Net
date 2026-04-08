"""
Улучшенный контроллер с правильной стартовой позицией и стабилизацией
"""
from vehicle import Car
from controller import Supervisor
import math
import random

class LearningController:
    def __init__(self):
        self.car = Car()
        self.supervisor = Supervisor()

        # Получаем ноды
        self.car_node = self.supervisor.getFromDef("CAR")
        self.goal_node = self.supervisor.getFromDef("GOAL")
        self.obstacle_node = self.supervisor.getFromDef("TRAINING_OBSTACLE")

        # Настраиваем сенсоры
        self.setup_sensors()

        # Параметры - ПОДНЯТАЯ стартовая позиция
        self.START_POS = [-12, 0, 0.5]  # Подняли повыше на 0.5
        self.START_ROT = [0, 1, 0, 0]

        # Параметры обучения
        self.GOAL_REWARD = 100
        self.COLLISION_PENALTY = -50
        self.STEP_PENALTY = -0.1
        self.COLLISION_DISTANCE = 3.0

        # Параметры стабилизации
        self.RESET_STEPS = 50  # Увеличили количество шагов для стабилизации
        self.RESET_SPEED = 0.0
        self.RESET_STEERING = 0.0

        # Переменные эпизода
        self.step_count = 0
        self.episode_count = 0
        self.episode_reward = 0
        self.success_count = 0
        self.collision_count = 0

        # Флаги состояния
        self.is_resetting = False
        self.reset_counter = 0

        print("\n" + "="*50)
        print("🤖 ОБУЧАЮЩИЙСЯ КОНТРОЛЛЕР")
        print("="*50)
        print(f"Старт: {self.START_POS}")
        print(f"Награда за цель: {self.GOAL_REWARD}")
        print(f"Штраф за столкновение: {self.COLLISION_PENALTY}")
        print(f"Шаги стабилизации: {self.RESET_STEPS}")
        print("="*50)

    def setup_sensors(self):
        self.lidar_front = self.car.getDevice("lidar_front")
        if self.lidar_front:
            self.lidar_front.enable(32)
            print("✅ Передний лидар включен")

        self.gps = self.car.getDevice("gps")
        if self.gps:
            self.gps.enable(32)
            print("✅ GPS включен")

    def get_lidar_data(self):
        if self.lidar_front:
            data = self.lidar_front.getRangeImage()
            if data and len(data) > 0:
                valid = [x for x in data if 0.1 < x < 100]
                if valid:
                    return min(valid)
        return 30

    def get_position(self):
        if self.gps:
            return self.gps.getValues()
        elif self.car_node:
            return self.car_node.getPosition()
        return [0, 0, 0]

    def get_distance_to_goal(self):
        if not self.goal_node:
            return 100
        car_pos = self.get_position()
        goal_pos = self.goal_node.getPosition()
        return math.sqrt(
            (car_pos[0] - goal_pos[0])**2 +
            (car_pos[2] - goal_pos[2])**2
        )

    def check_collision(self):
        if not self.obstacle_node:
            return False
        car_pos = self.get_position()
        obs_pos = self.obstacle_node.getPosition()
        dist = math.sqrt(
            (car_pos[0] - obs_pos[0])**2 +
            (car_pos[2] - obs_pos[2])**2
        )
        return dist < 2.5

    def calculate_reward(self, prev_dist, current_dist, collision, goal):
        reward = self.STEP_PENALTY

        # Награда за приближение к цели
        if current_dist < prev_dist:
            reward += 0.5
        else:
            reward -= 0.3

        # Большие события
        if collision:
            reward += self.COLLISION_PENALTY
            print(f"💥 ШТРАФ: {self.COLLISION_PENALTY}")
        elif goal:
            reward += self.GOAL_REWARD
            print(f"🎉 НАГРАДА: {self.GOAL_REWARD}")

        return reward

    def reset_episode(self, reason):
        """Полный сброс эпизода с гарантированной стабилизацией"""
        self.episode_count += 1
        if reason == "goal":
            self.success_count += 1
        elif reason == "collision":
            self.collision_count += 1

        print(f"\n{'='*50}")
        print(f"📊 ЭПИЗОД {self.episode_count} ЗАВЕРШЕН")
        print(f"   Причина: {reason}")
        print(f"   Награда: {self.episode_reward:.1f}")
        print(f"   Успехи: {self.success_count}/{self.episode_count}")
        print(f"{'='*50}\n")

        # 1. Сначала останавливаем машину
        self.car.setCruisingSpeed(0)
        self.car.setSteeringAngle(0)

        # Небольшая задержка чтобы остановиться
        for _ in range(10):
            self.supervisor.step(32)

        # 2. Телепортируем машину
        if self.car_node:
            # Принудительно отключаем физику на момент телепортации
            self.car_node.getField("translation").setSFVec3f(self.START_POS)
            self.car_node.getField("rotation").setSFRotation(self.START_ROT)

            # Несколько шагов для применения трансформации
            for _ in range(5):
                self.supervisor.step(32)

        # 3. Полная остановка и стабилизация
        print("   ⏳ Стабилизация машины...")

        # Удерживаем машину неподвижной
        self.car.setCruisingSpeed(0)
        self.car.setSteeringAngle(0)

        # Длинный цикл стабилизации
        stabilization_steps = self.RESET_STEPS
        for i in range(stabilization_steps):
            # Проверяем позицию и принудительно удерживаем если нужно
            current_pos = self.get_position()
            if abs(current_pos[0] - self.START_POS[0]) > 0.1 or \
               abs(current_pos[2] - self.START_POS[2]) > 0.1:
                # Если машина сместилась, возвращаем обратно
                self.car_node.getField("translation").setSFVec3f(self.START_POS)
                self.car_node.getField("rotation").setSFRotation(self.START_ROT)

            self.supervisor.step(32)

            # Прогресс-индикатор
            if i % 10 == 0:
                print(f"      Стабилизация: {i}/{stabilization_steps}", end='\r')

        print("   ✅ Стабилизация завершена!")

        # 4. Финальная проверка позиции
        final_pos = self.get_position()
        print(f"   📍 Финальная позиция: x={final_pos[0]:.2f}, z={final_pos[2]:.2f}")

        # Сбрасываем счетчики эпизода
        self.episode_reward = 0
        self.is_resetting = False

    def run(self):
        print("\n🚗 ЗАПУСК ОБУЧЕНИЯ...\n")

        # Первичная стабилизация при старте
        print("Первичная стабилизация...")
        self.car.setCruisingSpeed(0)
        self.car.setSteeringAngle(0)
        for _ in range(30):
            self.supervisor.step(32)
        print("Готов к работе!\n")

        prev_distance = self.get_distance_to_goal()

        while self.car.step() != -1:
            self.step_count += 1

            # Пропускаем обработку во время сброса
            if self.is_resetting:
                continue

            # Проверка состояния
            collision = self.check_collision()
            goal_reached = self.get_distance_to_goal() < 2.0

            # Текущее расстояние
            current_distance = self.get_distance_to_goal()

            # Вычисление награды
            reward = self.calculate_reward(
                prev_distance, current_distance,
                collision, goal_reached
            )
            self.episode_reward += reward
            prev_distance = current_distance

            # Данные с лидара
            front_dist = self.get_lidar_data()

            # Управление
            if front_dist < 3.0:  # Препятствие близко
                # Случайный поворот для объезда
                steering = random.choice([-0.4, 0.4])
                speed = 10
            else:
                # Движение прямо
                steering = 0
                speed = 20

            self.car.setSteeringAngle(steering)
            self.car.setCruisingSpeed(speed)

            # Завершение эпизода
            if collision:
                self.is_resetting = True
                self.reset_episode("collision")
                prev_distance = self.get_distance_to_goal()  # Сброс расстояния
            elif goal_reached:
                self.is_resetting = True
                self.reset_episode("goal")
                prev_distance = self.get_distance_to_goal()

            # Информация
            if self.step_count % 100 == 0:
                print(f"\n📊 Шаг {self.step_count}:")
                print(f"   Эпизод: {self.episode_count + 1}")
                print(f"   Награда эпизода: {self.episode_reward:.1f}")
                print(f"   Расстояние до цели: {current_distance:.1f}м")
                print(f"   Расстояние до препятствия: {front_dist:.1f}м")

if __name__ == "__main__":
    controller = LearningController()
    controller.run()