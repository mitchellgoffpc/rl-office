import cv2
import math
import numpy as np
import gymnasium as gym


class BodyEnvironment(gym.Env):
  def __init__(self):
    super().__init__()
    self.action_space = gym.spaces.Discrete(4)
    self.observation_space = gym.spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8)

    self.outer_size = 100
    self.inner_size = 50
    self.obs_size = 50
    self.movement_speed = 1
    self.rotation_speed = 5

    self.robot_pos = None
    self.robot_angle = None
    self.reset()

  def step(self, action):
    if action in (0, 1):  # Move forward or backward
      movement_speed = self.movement_speed * (1 if action == 0 else -1)
      new_pos = (self.robot_pos[0] + movement_speed * math.cos(math.radians(self.robot_angle)),
                 self.robot_pos[1] + movement_speed * math.sin(math.radians(self.robot_angle)))
      if self.check_collision(new_pos):
        return self.get_observation(), -100, False, {}
      self.robot_pos = new_pos
      return self.get_observation(), 1, False, {}
    elif action in (2, 3):  # Rotate right or left
      rotation_speed = self.rotation_speed * (1 if action == 3 else -1)
      self.robot_angle = (self.robot_angle + rotation_speed) % 360
      return self.get_observation(), -1, False, {}
    else:
      raise ValueError(f"Invalid action: {action}")

  def reset(self):
    while True:
      self.robot_pos = (np.random.uniform(0, self.outer_size), np.random.uniform(0, self.outer_size))
      if not self.check_collision(self.robot_pos):
        break
    self.robot_angle = np.random.uniform(0, 360)
    return self.get_observation()

  def get_observation(self):
    return np.array([self.calculate_distance_to_wall(self.robot_pos)], dtype=np.float32)


  # Helper functions
  def calculate_distance_to_wall(self, origin):
    direction = np.array([math.cos(math.radians(self.robot_angle)), math.sin(math.radians(self.robot_angle))])

    outer_intersections = [(self.outer_size - origin[i]) / direction[i] if direction[i] > 0 else
                           -origin[i] / direction[i] if direction[i] < 0 else np.inf for i in range(2)]

    center = (self.outer_size / 2, self.outer_size / 2)
    square_bounds = np.array([center[0] - self.inner_size / 2, center[1] - self.inner_size / 2,
                              center[0] + self.inner_size / 2, center[1] + self.inner_size / 2])
    
    tmin = (square_bounds[[0,1]] - origin) / direction
    tmax = (square_bounds[[2,3]] - origin) / direction

    if tmin[0] > tmax[0]: tmin[0], tmax[0] = tmax[0], tmin[0]
    if tmin[1] > tmax[1]: tmin[1], tmax[1] = tmax[1], tmin[1]

    if tmin[0] < tmax[1] and tmin[1] < tmax[0] and min(tmax) > 0:
        inner_intersection = max(tmin) if max(tmin) >= 0 else min(tmax)
    else:
        inner_intersection = float('inf')

    return min(*outer_intersections, inner_intersection)

  def check_collision(self, pos):
    if pos[0] < 0 or pos[0] > self.outer_size or pos[1] < 0 or pos[1] > self.outer_size:  # Outer square collision
      return True
    inner_square_start = (self.outer_size - self.inner_size) / 2
    inner_square_end = (self.outer_size + self.inner_size) / 2
    if inner_square_start < pos[0] < inner_square_end and inner_square_start < pos[1] < inner_square_end:  # Inner square collision
      return True
    return False

  
  # Rendering functions
  
  def get_image_observation(self):
    margin = int(self.outer_size / 2)
    img_size = self.outer_size + 2 * margin
    inner_margin = int((self.outer_size - self.inner_size) / 2)

    # Draw the environment
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    cv2.rectangle(img, (margin, margin), (img_size - margin, img_size - margin), (255, 255, 255), -1)
    cv2.rectangle(img, (margin + inner_margin, margin + inner_margin), 
                  (img_size - margin - inner_margin, img_size - margin - inner_margin), 
                  (0, 0, 0), -1)

    robot_pos_in_img = (int(self.robot_pos[0] + margin), int(self.robot_pos[1] + margin))
    cv2.circle(img, robot_pos_in_img, 20, (0, 0, 255), -1)
    
    # Shift, rotate, crop, and resize the image
    M = np.float32([[1, 0, self.outer_size/2 - self.robot_pos[0]], [0, 1, self.outer_size/2 - self.robot_pos[1]]])
    img = cv2.warpAffine(img, M, (img_size, img_size))
    M = cv2.getRotationMatrix2D((img_size//2, img_size//2), self.robot_angle+90, 1)
    img = cv2.warpAffine(img, M, (img_size, img_size))

    crop_margin = int(self.obs_size / 2)
    cropped_img = img[img_size//2 - crop_margin : img_size//2 + crop_margin,
                      img_size//2 - crop_margin : img_size//2 + crop_margin]

    return cv2.resize(cropped_img, tuple(self.observation_space.shape[:2]))
  
  def render(self):
    # Scale factors for rendering
    img_size = 256
    scale_factor = img_size / self.outer_size
    obs_scale_factor = 15

    # Compute the points for the outer and inner squares based on the scale factors
    outer_rect_start_point = (0, 0)
    outer_rect_end_point = (int(self.outer_size * scale_factor - 1), int(self.outer_size * scale_factor - 1))
    inner_rect_start_point = (int(scale_factor * (self.outer_size - self.inner_size) / 2), int(scale_factor * (self.outer_size - self.inner_size) / 2))
    inner_rect_end_point = (int(scale_factor * (self.outer_size + self.inner_size) / 2 - 1), int(scale_factor * (self.outer_size + self.inner_size) / 2 - 1))
    
    # Draw the environment
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    cv2.rectangle(img, outer_rect_start_point, outer_rect_end_point, (255, 255, 255), -1)
    cv2.rectangle(img, inner_rect_start_point, inner_rect_end_point, (0, 0, 0), -1)
    
    # Draw a triangle for the robot
    direction_vector = np.array([math.cos(math.radians(self.robot_angle)),
                                 math.sin(math.radians(self.robot_angle))])
    perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])
    triangle_center = np.array(self.robot_pos) * scale_factor
    triangle_vertices = np.int32([triangle_center - obs_scale_factor*direction_vector - obs_scale_factor*perpendicular_vector,
                                  triangle_center - obs_scale_factor*direction_vector + obs_scale_factor*perpendicular_vector,
                                  triangle_center + obs_scale_factor*direction_vector])
    cv2.fillPoly(img, [triangle_vertices], color=(0, 0, 255))

    return img



if __name__ == '__main__':
  import pygame
  pygame.init()
  screen = pygame.display.set_mode((256, 256))
  clock = pygame.time.Clock()
  font = pygame.font.Font(None, 16)

  env = BodyEnvironment()
  observation = env.reset()
  running = True

  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
        running = False

    keys = pygame.key.get_pressed()
    actions = {pygame.K_a: 2, pygame.K_d: 3, pygame.K_w: 0, pygame.K_s: 1}
    for key, action in actions.items():
      if keys[key]:
        observation, reward, done, _ = env.step(action)
        if done:
          env.reset()
        break

    # Draw environment
    screen.fill((255, 255, 255))
    img = env.render()
    img = np.transpose(img, (1, 0, 2))
    img = pygame.surfarray.make_surface(img)
    screen.blit(img, (0, 0))

    dist_to_wall = env.get_observation()[0]
    text = font.render(f"Distance to wall: {dist_to_wall:.2f}", True, (0, 0, 0))
    screen.blit(text, (5, 5))

    pygame.display.flip()
    clock.tick(60)

  pygame.quit()
