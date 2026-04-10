import numpy as np

def choice_ray(front_state):
    danger_rays_left = []
    danger_rays_front = []
    danger_rays_right = []
    for e, i in enumerate(front_state):
        if i < 15:
            if e <= 57/3:
                danger_rays_left.append(i)
            elif 57/3 < e <= 57/3*2:
                danger_rays_front.append(i)
            else:
                danger_rays_right.append(i)
    return danger_rays_left, danger_rays_front, danger_rays_right
            
def danger_level(danger_rays_left, danger_rays_front, danger_rays_right, speed, wheel_angle):
    rays = [danger_rays_left, danger_rays_front, danger_rays_right]
    wheel_angle = check_wheel_angle(wheel_angle)

    for e, i in enumerate(rays):
        if len(i) > 0:
            min_rays = (min(i) if min(i) > 0 else 0.1)/15
        else:
            min_rays = 15/15
        if e == 1:
            wheel_angle = abs((e + wheel_angle)/2)
        else:
            wheel_angle = abs((e - wheel_angle)/2)

        y = (len(i)/19)*wheel_angle  #19*(40/0.1)
        norm_y = np.clip(y, 0, 1)
        rays[e] = norm_y
    return rays

def check_wheel_angle(wheel_angle):
    if wheel_angle < -0.2:
        wheel_angle = 2
    elif -0.2 <= wheel_angle <= 0.2:
        wheel_angle = 1
    else:
        wheel_angle = 0
    return wheel_angle
