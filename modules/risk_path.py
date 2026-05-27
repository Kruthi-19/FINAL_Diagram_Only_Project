import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from heapq import heappush, heappop

RESULT_DIR = "static/results"


def generate_kde_heatmap(image_path, grid, zone_map, event_data):
    heat = np.zeros_like(grid, dtype=float)
    crowd = max(50, int(event_data.get("crowd", 300)))
    factor = crowd / 300

    for z in zone_map:
        y, x = z["center"]
        label = z["label"]

        if label == "stage":
            heat[y, x] += 14 * factor
        elif label == "entry":
            heat[y, x] += 7 * factor
        elif label == "exit":
            heat[y, x] += 4 * factor
        elif label == "helpdesk":
            heat[y, x] += 6 * factor

    density = gaussian_filter(heat, sigma=3)

    name = os.path.splitext(os.path.basename(image_path))[0]
    out = os.path.join(RESULT_DIR, f"{name}_03_kde_heatmap.png")

    plt.figure(figsize=(6, 5))
    plt.imshow(density, cmap="jet")
    plt.title("KDE Crowd Risk Heatmap")
    plt.colorbar(label="Risk Density")
    plt.axis("off")
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    return density, out


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(node, grid):
    r, c = node
    # 4-direction movement
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            if grid[nr, nc] == 0:
                yield (nr, nc)


def astar(grid, density, start, goal, risk_weight=4.0, bias=None):
    """
    True obstacle-aware A*.
    It never expands blocked cells where grid value is 1.
    Cost = distance + KDE crowd density cost + optional direction bias.
    """
    pq = []
    heappush(pq, (0, start))
    came = {}
    cost = {start: 0}

    while pq:
        _, current = heappop(pq)

        if current == goal:
            path = []
            cur = current
            while cur in came:
                path.append(cur)
                cur = came[cur]
            path.append(start)
            return path[::-1], round(float(cost[current]), 2)

        for nb in neighbors(current, grid):
            # Base distance + risk cost
            step_cost = 1 + density[nb] * risk_weight

            # Bias is used only to create different valid route preferences
            if bias == "left":
                step_cost += nb[1] * 0.015
            elif bias == "right":
                step_cost += (grid.shape[1] - nb[1]) * 0.015
            elif bias == "top":
                step_cost += nb[0] * 0.010
            elif bias == "bottom":
                step_cost += (grid.shape[0] - nb[0]) * 0.010
            elif bias == "center":
                step_cost += abs(nb[1] - grid.shape[1] // 2) * 0.012

            new_cost = cost[current] + step_cost

            if nb not in cost or new_cost < cost[nb]:
                cost[nb] = new_cost
                priority = new_cost + heuristic(nb, goal)
                heappush(pq, (priority, nb))
                came[nb] = current

    return [], 999999


def nearest_free_cell(point, grid):
    """
    If detected entry/exit falls on blocked area, move it to nearest free cell.
    """
    r, c = point
    r = max(0, min(grid.shape[0] - 1, r))
    c = max(0, min(grid.shape[1] - 1, c))

    if grid[r, c] == 0:
        return (r, c)

    for radius in range(1, max(grid.shape)):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] == 0:
                        return (nr, nc)

    return (r, c)


def get_start_and_exits(zone_map, grid):
    entries = [z["center"] for z in zone_map if z["label"] == "entry"]
    exits = [z["center"] for z in zone_map if z["label"] == "exit"]

    start = entries[0] if entries else (grid.shape[0] - 2, grid.shape[1] // 2)
    goals = exits if exits else [(grid.shape[0] // 2, 2), (grid.shape[0] // 2, grid.shape[1] - 3)]

    start = nearest_free_cell(start, grid)
    goals = [nearest_free_cell(g, grid) for g in goals]

    return start, goals


def generate_multiple_paths(image_path, grid, density, zone_map):
    """
    Generates multiple TRUE A* paths.
    Each path is obstacle-safe because A* expands only walkable grid cells.
    Different route preferences are created using small bias terms.
    """
    start, goals = get_start_and_exits(zone_map, grid)

    left_exit = min(goals, key=lambda g: g[1])
    right_exit = max(goals, key=lambda g: g[1])

    configs = [
        ("Path 1 - Left Safe Route", left_exit, 2.0, "left"),
        ("Path 2 - Left Low-Risk Route", left_exit, 5.0, "top"),
        ("Path 3 - Center Balanced Route", right_exit, 3.0, "center"),
        ("Path 4 - Right Safe Route", right_exit, 2.0, "right"),
        ("Path 5 - Right Low-Risk Route", right_exit, 5.0, "top"),
        ("Path 6 - Wide Alternative Route", right_exit, 4.0, "bottom"),
    ]

    path_options = []

    for name, goal, risk_weight, bias in configs:
        path, cost = astar(grid, density, start, goal, risk_weight=risk_weight, bias=bias)

        if path:
            path_options.append({
                "name": name,
                "goal": goal,
                "risk_weight": risk_weight,
                "path": path,
                "cost": cost,
                "length": len(path)
            })

    # fallback if no path was found
    if not path_options:
        return [], None, None

    best = min(path_options, key=lambda p: p["cost"])
    out = save_paths(image_path, grid, density, start, path_options, best)

    return path_options, best, out


def save_paths(image_path, grid, density, start, paths, best):
    name = os.path.splitext(os.path.basename(image_path))[0]
    out = os.path.join(RESULT_DIR, f"{name}_04_paths_comparison.png")

    plt.figure(figsize=(8, 7))
    plt.imshow(density, cmap="jet", alpha=0.72)

    # Obstacles clearly shown
    oy, ox = np.where(grid == 1)
    plt.scatter(ox, oy, c="black", s=25, label="Obstacles / Blocked Cells", zorder=5)

    colors = ["white", "yellow", "orange", "lime", "magenta", "cyan"]

    # Plot all valid A* paths
    for idx, p in enumerate(paths):
        py, px = zip(*p["path"])
        plt.plot(
            px, py,
            color=colors[idx % len(colors)],
            linewidth=2.6,
            alpha=0.95,
            label=f"{idx+1}. {p['name']} | Cost {p['cost']}",
            zorder=8
        )

    # Highlight best path with black border + white center
    py, px = zip(*best["path"])
    plt.plot(px, py, color="black", linewidth=6, alpha=0.90, zorder=9)
    plt.plot(px, py, color="white", linewidth=3.5, alpha=1.0, label=f"BEST PATH | Cost {best['cost']}", zorder=10)

    plt.scatter(start[1], start[0], c="lime", s=180, edgecolors="black", label="Entry", zorder=11)
    plt.scatter(best["goal"][1], best["goal"][0], c="cyan", s=180, edgecolors="black", label="Selected Exit", zorder=11)

    plt.title("A* Obstacle-Safe Multiple Path Comparison", fontsize=13, weight="bold")
    plt.gca().invert_yaxis()
    plt.legend(fontsize=7, loc="lower right", framealpha=0.92)
    plt.axis("off")
    plt.savefig(out, bbox_inches="tight", dpi=180)
    plt.close()

    return out
