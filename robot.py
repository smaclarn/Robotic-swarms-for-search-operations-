import numpy as np
import matplotlib.pyplot as plt
import random
import datetime

# Constants
NUM_ROBOTS, NUM_TARGETS = 4, 5
INITIAL_OBSTACLES = np.array([[3, 3], [7, 7], [5, 5], [2, 6], [8, 8], [4, 2], [1, 4], [6, 8]])
STEP_SIZE, AVOIDANCE_DISTANCE = 0.1, 0.6
TARGET_REACHED_THRESHOLD, FORMATION_KEEPING_WEIGHT = 0.5, 0.4
MAX_SPEED, VELOCITY_DECAY = 0.15, 0.85
COMMUNICATION_RADIUS, FORMATION_RECOVERY_THRESHOLD = 2.5, 2.0
OBSTACLE_REPULSION_STRENGTH, LEADERSHIP_CHANGE_THRESHOLD = 2.5, 1.2
LEADERSHIP_SCORE_MEMORY, TARGET_DETECTION_RADIUS = 5, 0.8
EXPLORATION_WEIGHT, FORMATION_RECOVERY_DELAY = 0.6, 15
SENSOR_LINE_SMOOTHING, OBSTACLE_SPEED = 0.7, 0.05
OBSTACLE_DIRECTION_CHANGE_PROB, OBSTACLE_BOUNDARY_BUFFER = 0.02, 0.5
MAX_SIMULATION_STEPS, INITIAL_POSITION = 5000, np.array([1, 1])

class DynamicObstacle:
    def __init__(self, pos):
        self.position = np.array(pos, dtype=float)
        self.velocity = np.random.uniform(-1, 1, 2)
        self.velocity = self.velocity / np.linalg.norm(self.velocity) * OBSTACLE_SPEED if np.linalg.norm(self.velocity) > 0 else self.velocity

    def update(self, obstacles):
        if random.random() < OBSTACLE_DIRECTION_CHANGE_PROB:
            new_dir = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(new_dir) > 0:
                self.velocity = new_dir / np.linalg.norm(new_dir) * OBSTACLE_SPEED
        next_pos = self.position + self.velocity
        if next_pos[0] < OBSTACLE_BOUNDARY_BUFFER or next_pos[0] > 10 - OBSTACLE_BOUNDARY_BUFFER:
            self.velocity[0] *= -1
        if next_pos[1] < OBSTACLE_BOUNDARY_BUFFER or next_pos[1] > 10 - OBSTACLE_BOUNDARY_BUFFER:
            self.velocity[1] *= -1
        for obs in obstacles:
            if obs is not self and np.linalg.norm(next_pos - obs.position) < AVOIDANCE_DISTANCE * 2:
                repulsion = self.position - obs.position
                if np.linalg.norm(repulsion) > 0:
                    self.velocity += repulsion / np.linalg.norm(repulsion) * 0.01
        if np.linalg.norm(self.velocity) > 0:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * OBSTACLE_SPEED
        self.position += self.velocity

class Robot:
    def __init__(self, pos, offset, id):
        self.position, self.velocity = np.array(pos, dtype=float), np.zeros(2, dtype=float)
        self.formation_offset, self.id = offset, id
        self.discovered_targets, self.known_targets = set(), set()
        self.formation_deviation, self.avoidance_mode = 0.0, False
        self.leadership_score, self.score_history = 0.0, [0.0] * LEADERSHIP_SCORE_MEMORY
        self.is_leader, self.visited_positions = False, set()
        self.exploration_target, self.exploration_time = None, 0
        self.sensor_line_points, self.formation_recovery_timer = [], 0
        self.initial_position = pos

    def calculate_leadership_score(self, obstacles):
        min_obstacle_dist = min(np.linalg.norm(self.position - obs.position) for obs in obstacles)
        obstacle_factor = np.clip(min_obstacle_dist / 5.0, 0, 1)
        coverage_factor = min(1.0, len(self.visited_positions) / 100.0)
        speed_stability = 1.0 - min(1.0, np.linalg.norm(self.velocity) / MAX_SPEED)
        center_dist = np.linalg.norm(self.position - np.array([5.0, 5.0]))
        center_factor = 1.0 - np.clip(center_dist / 7.0, 0, 1)
        discovery_factor = min(1.0, len(self.discovered_targets) / NUM_TARGETS)
        score = obstacle_factor * 2.0 + coverage_factor * 2.0 + speed_stability * 1.0 + center_factor * 0.5 + discovery_factor * 2.5
        self.score_history.pop(0)
        self.score_history.append(score)
        self.leadership_score = sum(self.score_history) / len(self.score_history)
        return self.leadership_score

    def choose_exploration_target(self):
        while True:
            target = np.array([random.uniform(1, 9), random.uniform(1, 9)])
            if all(np.linalg.norm(np.array(pos) - target) >= 1.0 for pos in self.visited_positions):
                return target

    def move_towards_target(self, target):
        direction = target - self.position
        distance = np.linalg.norm(direction)
        if distance > 0:
            self.velocity += (direction / distance) * STEP_SIZE * 1.1
        if self.exploration_target is not None:
            if not self.sensor_line_points:
                self.sensor_line_points = [self.position.copy(), self.position * 0.7 + target * 0.3, self.position * 0.3 + target * 0.7, target.copy()]
            else:
                self.sensor_line_points[0] = self.position.copy()
                mid1 = self.position * (1 - SENSOR_LINE_SMOOTHING) + self.sensor_line_points[1] * SENSOR_LINE_SMOOTHING
                mid2 = mid1 * (1 - SENSOR_LINE_SMOOTHING) + target * SENSOR_LINE_SMOOTHING
                self.sensor_line_points[1], self.sensor_line_points[2], self.sensor_line_points[3] = mid1, mid2, target.copy()

    def avoid_obstacles(self, obstacles):
        avoidance_force = np.zeros(2, dtype=float)
        was_avoiding = self.avoidance_mode
        self.avoidance_mode = False
        for obs in obstacles:
            direction = self.position - obs.position
            distance = np.linalg.norm(direction)
            if 0 < distance < AVOIDANCE_DISTANCE:
                repulsion_strength = OBSTACLE_REPULSION_STRENGTH * (1 - (distance / AVOIDANCE_DISTANCE) ** 3)
                self.avoidance_mode = True
                if distance < AVOIDANCE_DISTANCE * 0.8:
                    repulsion_strength *= 2.0
                if distance > 0:
                    avoidance_force += (direction / distance) * repulsion_strength
        next_position = self.position + self.velocity
        for obs in obstacles:
            if np.linalg.norm(next_position - obs.position) < AVOIDANCE_DISTANCE:
                self.velocity *= 0.3
                break
        self.velocity += avoidance_force
        if self.avoidance_mode != was_avoiding:
            self.formation_recovery_timer = FORMATION_RECOVERY_DELAY

    def maintain_formation(self, leader):
        if leader and not self.is_leader:
            target_position = leader.position + self.formation_offset
            formation_error = target_position - self.position
            self.formation_deviation = np.linalg.norm(formation_error)
            if self.formation_recovery_timer > 0:
                self.formation_recovery_timer -= 1
                recovery_factor = 1.0 - (self.formation_recovery_timer / FORMATION_RECOVERY_DELAY)
                formation_force = formation_error * (FORMATION_KEEPING_WEIGHT * 0.2 * recovery_factor)
            elif self.avoidance_mode:
                formation_force = formation_error * (FORMATION_KEEPING_WEIGHT * 0.1)
            elif self.formation_deviation > FORMATION_RECOVERY_THRESHOLD:
                formation_force = formation_error * (FORMATION_KEEPING_WEIGHT * 1.8)
            else:
                formation_force = formation_error * FORMATION_KEEPING_WEIGHT
            self.velocity += formation_force

    def avoid_other_robots(self, robots):
        separation_force = np.zeros(2, dtype=float)
        for other in robots:
            if other is not self:
                direction = self.position - other.position
                distance = np.linalg.norm(direction)
                if 0 < distance < 0.7:
                    separation_strength = 0.8 * (0.7 - distance) ** 2
                    separation_force += (direction / distance) * separation_strength
        self.velocity += separation_force

    def detect_targets(self, all_targets):
        for target in all_targets:
            target_tuple = tuple(target)
            if target_tuple not in self.known_targets and np.linalg.norm(self.position - target) <= TARGET_DETECTION_RADIUS:
                self.discovered_targets.add(target_tuple)
                self.known_targets.add(target_tuple)
                return target_tuple
        return None

    def share_target_info(self, robots):
        for other in robots:
            if other is not self and np.linalg.norm(self.position - other.position) <= COMMUNICATION_RADIUS:
                other.known_targets.update(self.known_targets)

    def update(self, robots, leader, all_targets, obstacles):
        current_pos = (round(self.position[0], 1), round(self.position[1], 1))
        self.visited_positions.add(current_pos)
        if len(self.visited_positions) > 200:
            self.visited_positions.pop()
        new_target = self.detect_targets(all_targets)
        self.share_target_info(robots)
        if self.is_leader:
            known_targets = [t for t in all_targets if tuple(t) in self.known_targets and tuple(t) not in [tuple(target) for target in all_targets if any(tuple(target) in r.discovered_targets for r in robots)]]
            all_discovered = set().union(*[r.discovered_targets for r in robots])
            known_targets = [t for t in known_targets if tuple(t) not in all_discovered]
            if known_targets:
                closest_target = min(known_targets, key=lambda t: np.linalg.norm(self.position - t))
                self.move_towards_target(closest_target)
                self.exploration_target, self.sensor_line_points = None, []
            else:
                if self.exploration_target is None or self.exploration_time <= 0 or np.linalg.norm(self.position - self.exploration_target) < 0.5:
                    self.exploration_target = self.choose_exploration_target()
                    self.exploration_time = random.randint(30, 50)
                self.exploration_time -= 1
                self.move_towards_target(self.exploration_target)
            for target in all_targets:
                if np.linalg.norm(self.position - target) < TARGET_REACHED_THRESHOLD:
                    self.discovered_targets.add(tuple(target))
                    self.known_targets.add(tuple(target))
                    if self.exploration_target is not None:
                        self.exploration_target, self.exploration_time, self.sensor_line_points = None, 0, []
        self.avoid_obstacles(obstacles)
        self.avoid_other_robots(robots)
        self.maintain_formation(leader)
        next_position = self.position + self.velocity
        for obs in obstacles:
            if np.linalg.norm(next_position - obs.position) < AVOIDANCE_DISTANCE:
                obs_direction = next_position - obs.position
                if np.linalg.norm(obs_direction) > 0:
                    safe_position = obs.position + (obs_direction / np.linalg.norm(obs_direction)) * AVOIDANCE_DISTANCE
                    self.velocity = (safe_position - self.position) * 0.9
                    break
        speed = np.linalg.norm(self.velocity)
        if speed > MAX_SPEED:
            self.velocity = (self.velocity / speed) * MAX_SPEED
        self.position += self.velocity
        self.velocity *= VELOCITY_DECAY
        if self.is_leader and self.exploration_target is not None and np.linalg.norm(self.position - self.exploration_target) < 0.3:
            self.sensor_line_points = []

def update_leadership(robots, obstacles, targets):
    if not targets:
        return None
    for robot in robots:
        robot.calculate_leadership_score(obstacles)
    all_discovered = set().union(*[r.discovered_targets for r in robots])
    remaining_targets = [t for t in targets if tuple(t) not in all_discovered]
    if remaining_targets:
        robots_with_target_knowledge = [r for r in robots if any(tuple(t) in r.known_targets for t in remaining_targets)]
        potential_leaders = robots_with_target_knowledge if robots_with_target_knowledge else robots
    else:
        potential_leaders = robots
    new_leader = max(potential_leaders, key=lambda r: r.leadership_score)
    for robot in robots:
        robot.is_leader = (robot.id == new_leader.id)
    return new_leader

def generate_random_targets(num_targets, obstacles, grid_size=10):
    targets = []
    for _ in range(num_targets):
        while True:
            candidate = np.array([random.uniform(0.5, grid_size - 0.5), random.uniform(0.5, grid_size - 0.5)])
            if min(np.linalg.norm(candidate - obs.position) for obs in obstacles) > 1.0:
                targets.append(candidate)
                break
    return targets

def visualize_swarm():
    obstacles = [DynamicObstacle(pos) for pos in INITIAL_OBSTACLES]
    targets = generate_random_targets(NUM_TARGETS, obstacles)
    offsets = [np.array([0.5, 0.5]), np.array([-0.5, 0.5]), np.array([-0.5, -0.5]), np.array([0.5, -0.5])]
    robots = [Robot(INITIAL_POSITION + offset, offset, i) for i, offset in enumerate(offsets)]
    robots[0].is_leader = True
    leader = robots[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    trail_history = {i: [] for i in range(NUM_ROBOTS)}
    obstacle_trail_history = {i: [] for i in range(len(obstacles))}
    max_trail_length, leadership_changes, collision_detected = 20, 0, False
    all_discovered_targets, start_time = set(), datetime.datetime.now()
    targets_all_discovered = False

    for step in range(MAX_SIMULATION_STEPS):
        ax.clear()
        for obs in obstacles:
            obs.update(obstacles)
            obstacle_trail_history[obstacles.index(obs)].append(obs.position.copy())
            if len(obstacle_trail_history[obstacles.index(obs)]) > max_trail_length:
                obstacle_trail_history[obstacles.index(obs)].pop(0)
        previous_leader = leader
        new_leader = update_leadership(robots, obstacles, targets)
        if new_leader is not leader:
            leadership_changes += 1
            leader = new_leader
            if previous_leader and new_leader and previous_leader.id != new_leader.id:
                new_leader.exploration_target, new_leader.exploration_time, new_leader.sensor_line_points = None, 0, []
        for i, obs in enumerate(obstacles):
            if len(obstacle_trail_history[i]) > 1:
                trail = np.array(obstacle_trail_history[i])
                ax.plot(trail[:, 0], trail[:, 1], 'gray', alpha=0.2)
            ax.scatter(obs.position[0], obs.position[1], c='black', marker='s', s=150)
            circle = plt.Circle(obs.position, AVOIDANCE_DISTANCE, color='gray', fill=False, linestyle='--', alpha=0.3)
            ax.add_patch(circle)
        all_discovered_targets.clear()
        for robot in robots:
            all_discovered_targets.update(robot.discovered_targets)
        for i, robot in enumerate(robots):
            robot.update(robots, leader, targets, obstacles)
            for obs in obstacles:
                if np.linalg.norm(robot.position - obs.position) < AVOIDANCE_DISTANCE:
                    collision_detected = True
                    print(f"Robot {i} inside avoidance zone at step {step}!")
            trail_history[i].append(robot.position.copy())
            if len(trail_history[i]) > max_trail_length:
                trail_history[i].pop(0)
            if len(trail_history[i]) > 1:
                trail = np.array(trail_history[i])
                ax.plot(trail[:, 0], trail[:, 1], 'b-', alpha=0.3)
            color = 'red' if robot.is_leader else 'blue'
            ax.scatter(*robot.position, c=color, s=100)
            ax.text(robot.position[0], robot.position[1] + 0.2, f"R{robot.id}", ha='center', va='center', fontsize=8)
            ax.text(robot.position[0], robot.position[1] - 0.2, f"{robot.leadership_score:.1f}", ha='center', va='center', fontsize=7)
            if not robot.is_leader and leader:
                ax.plot([leader.position[0], robot.position[0]], [leader.position[1], robot.position[1]], 'k--', alpha=0.3)
            if robot.is_leader and robot.exploration_target is not None:
                ax.scatter(*robot.exploration_target, c='yellow', marker='*', s=100, alpha=0.5)
                if robot.sensor_line_points and len(robot.sensor_line_points) >= 4 and not any(np.linalg.norm(robot.position - target) < TARGET_DETECTION_RADIUS * 1.5 for target in targets):
                    t = np.linspace(0, 1, 50)
                    points = np.array(robot.sensor_line_points)
                    curve_points = [(1 - ti) ** 3 * points[0] + 3 * (1 - ti) ** 2 * ti * points[1] + 3 * (1 - ti) * ti ** 2 * points[2] + ti ** 3 * points[3] for ti in t]
                    curve_points = np.array(curve_points)
                    ax.plot(curve_points[:, 0], curve_points[:, 1], 'y-', alpha=0.7)
            detection_circle = plt.Circle(robot.position, TARGET_DETECTION_RADIUS, color='green', fill=False, linestyle=':', alpha=0.2)
            ax.add_patch(detection_circle)
        for target in targets:
            color = 'green' if tuple(target) in all_discovered_targets else 'red'
            ax.scatter(*target, c=color, marker='X', s=200)
        ax.scatter(INITIAL_POSITION[0], INITIAL_POSITION[1], c='purple', marker='^', s=200, label='Initial Position')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        title = f"Step {step + 1}: Targets {len(all_discovered_targets)}/{NUM_TARGETS} - Leader: R{leader.id} - Changes: {leadership_changes}"
        if collision_detected:
            title += " - COLLISION DETECTED!"
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        if len(all_discovered_targets) == NUM_TARGETS:
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            targets_all_discovered = True
            print(f"All {NUM_TARGETS} targets discovered at step {step} in {duration:.1f} seconds! Stopping simulation.")
            plt.text(5, 5, "All targets discovered!", horizontalalignment='center', fontsize=16, bbox=dict(facecolor='white', alpha=0.8))
            plt.pause(2)
            break
        plt.pause(0.1)

    if targets_all_discovered:
        for step in range(MAX_SIMULATION_STEPS):
            ax.clear()
            for obs in obstacles:
                obs.update(obstacles)
                obstacle_trail_history[obstacles.index(obs)].append(obs.position.copy())
                if len(obstacle_trail_history[obstacles.index(obs)]) > max_trail_length:
                    obstacle_trail_history[obstacles.index(obs)].pop(0)
            for i, obs in enumerate(obstacles):
                if len(obstacle_trail_history[i]) > 1:
                    trail = np.array(obstacle_trail_history[i])
                    ax.plot(trail[:, 0], trail[:, 1], 'gray', alpha=0.2)
                ax.scatter(obs.position[0], obs.position[1], c='black', marker='s', s=150)
                circle = plt.Circle(obs.position, AVOIDANCE_DISTANCE, color='gray', fill=False, linestyle='--', alpha=0.3)
                ax.add_patch(circle)
            for i, robot in enumerate(robots):
                direction = robot.initial_position - robot.position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    robot.velocity += (direction / distance) * STEP_SIZE * 1.1
                robot.avoid_obstacles(obstacles)
                robot.avoid_other_robots(robots)
                speed = np.linalg.norm(robot.velocity)
                if speed > MAX_SPEED:
                    robot.velocity = (robot.velocity / speed) * MAX_SPEED
                robot.position += robot.velocity
                robot.velocity *= VELOCITY_DECAY
                trail_history[i].append(robot.position.copy())
                if len(trail_history[i]) > max_trail_length:
                    trail_history[i].pop(0)
                if len(trail_history[i]) > 1:
                    trail = np.array(trail_history[i])
                    ax.plot(trail[:, 0], trail[:, 1], 'b-', alpha=0.3)
                color = 'red' if robot.is_leader else 'blue'
                ax.scatter(*robot.position, c=color, s=100)
                ax.text(robot.position[0], robot.position[1] + 0.2, f"R{robot.id}", ha='center', va='center', fontsize=8)
                ax.text(robot.position[0], robot.position[1] - 0.2, f"{robot.leadership_score:.1f}", ha='center', va='center', fontsize=7)
            for target in targets:
                color = 'green' if tuple(target) in all_discovered_targets else 'red'
                ax.scatter(*target, c=color, marker='X', s=200)
            ax.scatter(INITIAL_POSITION[0], INITIAL_POSITION[1], c='purple', marker='^', s=200, label='Initial Position')
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            title = f"Returning to initial position: Step {step + 1}"
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.7)
            if all(np.linalg.norm(robot.position - robot.initial_position) < 0.1 for robot in robots):
                plt.text(5, 5, "All robots returned home!", horizontalalignment='center', fontsize=16, bbox=dict(facecolor='white', alpha=0.8))
                plt.pause(2)
                break
            plt.pause(0.1)

    if not targets_all_discovered:
        print(f"Simulation ended after {MAX_SIMULATION_STEPS} steps.")
        print(f"Only {len(all_discovered_targets)}/{NUM_TARGETS} targets were discovered.")
        plt.text(5, 5, f"Simulation timeout: {len(all_discovered_targets)}/{NUM_TARGETS} targets found", horizontalalignment='center', fontsize=16, bbox=dict(facecolor='white', alpha=0.8))
        plt.pause(2)

    plt.show()

if __name__ == "__main__":
    visualize_swarm()