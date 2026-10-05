#Antek Singer 
#10/01/2026

import heapq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class AStar3DPlanner:
    def __init__(self, grid, resolution, origin):
        self.grid = grid
        self.resolution = resolution
        self.origin = np.array(origin)
        self.shape = grid.shape
        
        # Pre-compute the 26-connectivity neighbor displacements and costs
        self.neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    cost = np.sqrt(dx**2 + dy**2 + dz**2) * self.resolution
                    self.neighbors.append((dx, dy, dz, cost))

    def world_to_grid(self, world_pos):
        grid_pos = np.round((np.array(world_pos) - self.origin) / self.resolution).astype(int)
        return tuple(np.clip(grid_pos, 0, np.array(self.shape) - 1))

    def grid_to_world(self, grid_pos):
        return self.origin + np.array(grid_pos) * self.resolution

    def _heuristic(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2)) * self.resolution

    def plan(self, start_world, goal_world):
        start_grid = self.world_to_grid(start_world)
        goal_grid = self.world_to_grid(goal_world)

        if self.grid[start_grid] == 1 or self.grid[goal_grid] == 1:
            print("Error: Start or Goal is inside an obstacle!")
            return None

        open_set = []
        heapq.heappush(open_set, (self._heuristic(start_grid, goal_grid), 0, start_grid))
        
        came_from = {}
        g_score = {start_grid: 0}
        closed_set = set()

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == goal_grid:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_grid)
                return path[::-1]

            if current in closed_set:
                continue
            closed_set.add(current)

            for dx, dy, dz, step_cost in self.neighbors:
                nx, ny, nz = current[0] + dx, current[1] + dy, current[2] + dz

                # FIXED: Proper individual index element checks
                if (0 <= nx < self.shape[0] and 
                    0 <= ny < self.shape[1] and 
                    0 <= nz < self.shape[2]):
                    
                    neighbor = (nx, ny, nz)
                    if self.grid[neighbor] == 1:
                        continue

                    tentative_g = current_g + step_cost

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + self._heuristic(neighbor, goal_grid)
                        came_from[neighbor] = current
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        print("A* failed to find a valid path.")
        return None

    def check_line_of_sight(self, p1_grid, p2_grid):
        p1 = np.array(p1_grid)
        p2 = np.array(p2_grid)
        distance = np.linalg.norm(p2 - p1)
        
        if distance == 0:
            return True
            
        steps = int(np.ceil(distance * 2))
        t_values = np.linspace(0, 1, steps)
        
        for t in t_values:
            sample = np.round(p1 + t * (p2 - p1)).astype(int)
            
            # FIXED: Proper discrete element checks
            if (sample[0] < 0 or sample[0] >= self.shape[0] or
                sample[1] < 0 or sample[1] >= self.shape[1] or
                sample[2] < 0 or sample[2] >= self.shape[2]):
                return False
            
            if self.grid[tuple(sample)] == 1:
                return False
        return True

    def prune_path(self, grid_path):
        if not grid_path:
            return []

        pruned_world_waypoints = [self.grid_to_world(grid_path[0])]
        start_idx = 0
        
        while start_idx < len(grid_path) - 1:
            los_found = False
            for next_idx in range(len(grid_path) - 1, start_idx, -1):
                if self.check_line_of_sight(grid_path[start_idx], grid_path[next_idx]):
                    pruned_world_waypoints.append(self.grid_to_world(grid_path[next_idx]))
                    start_idx = next_idx
                    los_found = True
                    break
                    
            if not los_found:
                start_idx += 1
                pruned_world_waypoints.append(self.grid_to_world(grid_path[start_idx]))
                    
        return pruned_world_waypoints

def visualize_3d_paths(grid, resolution, origin, raw_path, pruned_waypoints):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    filled_indices = np.argwhere(grid == 1)
    if len(filled_indices) > 0:
        x_obs = origin[0] + filled_indices[:, 0] * resolution
        y_obs = origin[1] + filled_indices[:, 1] * resolution
        z_obs = origin[2] + filled_indices[:, 2] * resolution
        ax.scatter(x_obs, y_obs, z_obs, color='red', alpha=0.05, marker='s', s=40, label='Obstacles')

    raw_world_path = np.array([np.array(origin) + np.array(node) * resolution for node in raw_path])
    ax.plot(raw_world_path[:, 0], raw_world_path[:, 1], raw_world_path[:, 2], 
            color='blue', linestyle='--', linewidth=2, label='Raw A* Path')

    pruned_pts = np.array(pruned_waypoints)
    ax.plot(pruned_pts[:, 0], pruned_pts[:, 1], pruned_pts[:, 2], 
            color='green', linestyle='-', linewidth=3, label='Pruned Waypoints')
    ax.scatter(pruned_pts[:, 0], pruned_pts[:, 1], pruned_pts[:, 2], 
               color='darkgreen', s=80, marker='o', edgecolors='black', label='Keyframe Waypoints')

    ax.set_title("3D A* Path Smoothing Pipeline for Isaac Sim", fontsize=12, fontweight='bold')
    ax.set_xlabel("X (Meters)")
    ax.set_ylabel("Y (Meters)")
    ax.set_zlabel("Z (Meters)")
    ax.set_xlim(0, grid.shape[0] * resolution)
    ax.set_ylim(0, grid.shape[1] * resolution)
    ax.set_zlim(0, grid.shape[2] * resolution)
    ax.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    workspace_grid = np.zeros((40, 40, 20), dtype=int)
    workspace_grid[20:22, 0:30, :] = 1  # Wall blocks up to Y=15 meters (index 30)

    planner = AStar3DPlanner(grid=workspace_grid, resolution=0.5, origin=(0.0, 0.0, 0.0))

    start_world = (2.0, 2.0, 1.0)
    goal_world  = (15.0, 2.0, 5.0)

    raw_grid_path = planner.plan(start_world, goal_world)

    if raw_grid_path:
        smooth_waypoints = planner.prune_path(raw_grid_path)
        visualize_3d_paths(workspace_grid, 0.5, (0.0, 0.0, 0.0), raw_grid_path, smooth_waypoints)
