"""从地图生成输入路线。这里只搜索二维自由栅格，不进行车辆状态搜索。"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def world_to_grid(points, map_data):
    # 第1步：把世界坐标转到地图局部坐标，成功后得到 [列坐标, 行坐标]。
    delta = np.asarray(points, dtype=np.float64) - map_data["origin"]
    yaw = float(map_data["origin_yaw"])
    grid_x = np.cos(yaw) * delta[..., 0] + np.sin(yaw) * delta[..., 1]
    grid_y = -np.sin(yaw) * delta[..., 0] + np.cos(yaw) * delta[..., 1]
    return np.stack([grid_x, grid_y], axis=-1) / float(map_data["resolution"])


def grid_to_world(points, map_data):
    points = np.asarray(points) * float(map_data["resolution"])
    yaw = float(map_data["origin_yaw"])
    x = np.cos(yaw) * points[..., 0] - np.sin(yaw) * points[..., 1]
    y = np.sin(yaw) * points[..., 0] + np.cos(yaw) * points[..., 1]
    return np.stack([x, y], axis=-1) + map_data["origin"]


class CoarseRoutePlanner:
    def __init__(self, map_data):
        # 第2步：把自由栅格和骨架组成稀疏图；靠近骨架、障碍间隙大的边代价较低。
        # 图只与地图有关，同一地图可以重复使用，训练时不读取任何参考轨迹。
        self.map_data = map_data
        features = map_data["map_features"]
        self.height, self.width = features.shape[1:]
        self.free = features[0] < 0.5
        row_ids, col_ids = np.indices(self.free.shape)
        node_ids = row_ids * self.width + col_ids
        sources = []
        targets = []
        costs = []
        for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            next_rows = row_ids + dy
            next_cols = col_ids + dx
            valid = (next_rows >= 0) & (next_rows < self.height)
            valid = valid & (next_cols >= 0) & (next_cols < self.width)
            rows = row_ids[valid]
            cols = col_ids[valid]
            nr = next_rows[valid]
            nc = next_cols[valid]
            allowed = self.free[rows, cols] & self.free[nr, nc]
            if dx != 0 and dy != 0:
                # 防止对角线从两个相邻障碍之间穿角。
                allowed = allowed & self.free[rows, nc] & self.free[nr, cols]
            rows, cols, nr, nc = rows[allowed], cols[allowed], nr[allowed], nc[allowed]
            clearance = np.minimum(features[1, rows, cols], features[1, nr, nc])
            skeleton = np.maximum(features[2, rows, cols], features[2, nr, nc])
            cost = np.hypot(dx, dy) * (1.0 + 0.5 / (clearance + 0.5) + 0.5 * (1.0 - skeleton))
            sources.extend([node_ids[rows, cols], node_ids[nr, nc]])
            targets.extend([node_ids[nr, nc], node_ids[rows, cols]])
            costs.extend([cost, cost])
        self.graph = csr_matrix(
            (np.concatenate(costs), (np.concatenate(sources), np.concatenate(targets))),
            shape=(self.height * self.width, self.height * self.width),
        )

    def plan(self, start_state, goal_state):
        # 第3步：只用起终位置查询粗路线，成功后得到变长世界坐标 [L,2]。
        # 路线只说明通道连接关系；车辆是否能通过，仍由后续完整车体检查决定。
        endpoints = world_to_grid(np.asarray([start_state[:2], goal_state[:2]]), self.map_data)
        cells = np.floor(endpoints).astype(np.int64)
        for x, y in cells:
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                raise ValueError("起点或终点位于地图外")
            if not self.free[y, x]:
                raise ValueError("起点或终点的前桥中心位于障碍物内")
        source = cells[0, 1] * self.width + cells[0, 0]
        target = cells[1, 1] * self.width + cells[1, 0]
        distances, predecessors = dijkstra(self.graph, indices=source, return_predecessors=True)
        if not np.isfinite(distances[target]):
            raise ValueError("起点和终点之间没有连通的自由栅格路线")
        nodes = [int(target)]
        while nodes[-1] != source:
            nodes.append(int(predecessors[nodes[-1]]))
        nodes.reverse()
        # 每4格保留一个实际路线点，复用原来的局部地图裁剪结构。
        selected = nodes[::4]
        if selected[-1] != target:
            selected.append(int(target))
        route_grid = []
        for node in selected:
            route_grid.append([node % self.width, node // self.width])
        route = grid_to_world(route_grid, self.map_data)
        if len(route) == 1:
            route = np.repeat(route, 2, axis=0)
        route[0] = start_state[:2]
        route[-1] = goal_state[:2]
        return route.astype(np.float32)
