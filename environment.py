from utils import *
from scipy.spatial.distance import cdist, pdist


class Environment:
    def __init__(self,
                 env_mode: str,
                 ):
        self.env_mode = env_mode

        # INPUTS
        self.num_workers = 2

        # X, Y COORDINATES
        self.worker_coords_init = np.array([
            [0, 0],
            [2, 1]
        ])
        # self.demand_map_init = np.array([
        #     [3, 0, 1, 1],
        #     [0, 2, 2, 1],
        #     [1, 2, 3, 1],
        #     [0, 3, 3, 3],
        #     [0, 0, 2, 1],
        # ])
        # self.altitude_map_init = np.array([
        #     [0, 0, 0, 0],
        #     [0, 0, 0, 0],
        #     [0, 0, 0, 0],
        #     [0, 0, 0, 0],
        #     [0, 0, 0, 0],
        # ])
        self.demand_map_init = np.array([
            [2, 0, 1],
            [0, 1, 2],
        ])
        self.altitude_map_init = np.array([
            [0, 0, 0],
            [0, 0, 0],
        ])

        # HEIGHT -> ROWS
        # WIDTH -> COLS
        # -------------------------------------------------------------------------------------------------------------
        self.table, self.height, self.width, self.cell_ids, self.worker_ids, self.cell_coords = None, None, None, None, None, None
        self.t2t = None
        self.w2t = None
        self.w2w = None

        self.visited_cell_history = None
        self.trajectories = []
        self.demand_history = []
        self.altitude_history = []
        self.coord_col_names = ['x', 'y', 'z']
        self.demand_map = deepcopy(self.demand_map_init)
        self.altitude_map = deepcopy(self.altitude_map_init)
        self.num_cells = self.demand_map_init.sum()
        self.num_completed_cells = 0

        self.env_ticks = 0
        self.initialize()
        # CELL is a single soil unit
        # -------------------------------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------

    def initialize(self):
        if self.env_mode == 'train':
            # TODO: create env generation methods
            self.table = self.construct_table()
        self.construct_distance_matrices()

    def construct_table(self):
        self.height, self.width = self.demand_map_init.shape
        table = -np.ones([self.num_cells + self.num_workers, n_cols_table], dtype='float')
        self.cell_ids = np.arange(self.num_cells)
        self.worker_ids = np.arange(self.num_workers) + self.num_cells

        cell_id = 0

        num_rows, num_cols = self.height, self.width
        # -------------------------------------------------------------------------------------------------------------
        # IMPUTE CELL COORDS
        table[:, TC['accessible']] = 0
        table[:, TC['has_ug']] = 0
        for r in range(num_rows):
            for c in range(num_cols):
                if self.demand_map_init[r, c] == 0:
                    continue
                else:
                    depth = self.demand_map_init[r, c]
                    for d in range(depth):
                        table[cell_id, TC['env_id']] = cell_id
                        table[cell_id, TC['row']] = r
                        table[cell_id, TC['col']] = c
                        table[cell_id, get_cols(self.coord_col_names, TC)] = c, num_rows - r - 1, \
                                                                             self.altitude_map_init[r, c] - (d + 1)
                        if d == 0:  # mark cell as surface (accessible)
                            table[cell_id, TC['accessible']] = 1
                        if d + 1 < depth:
                            table[cell_id, TC['has_ug']] = 1
                        cell_id += 1
        # -------------------------------------------------------------------------------------------------------------
        # IMPUTE WORKER COORDS
        for w, worker_id in enumerate(self.worker_ids):
            x, y = self.worker_coords_init[w][0], self.worker_coords_init[w][1]
            table[worker_id, TC['env_id']] = worker_id
            table[worker_id, get_cols(self.coord_col_names, TC)] = x, y, self.altitude_map_init[y, x]

        # coords = self.get_coords(table[:, TC['env_id']].astype(int))

        table[:, TC['active']] = 1
        table[:, TC['assigned']] = 0
        table[self.cell_ids, TC['soil']] = 1
        table[self.cell_ids, TC['worker']] = 0
        table[self.cell_ids, TC['process_time']] = 0

        table[self.worker_ids, TC['soil']] = 0
        table[self.worker_ids, TC['worker']] = 1
        table[self.worker_ids, TC['accessible']] = 1

        self.visited_cell_history = dict((i, []) for i in self.worker_ids)
        self.trajectories.append((self.env_ticks, self.worker_coords_init))  # add to trajectory
        self.demand_history.append((self.env_ticks, self.demand_map_init))
        self.altitude_history.append((self.env_ticks, self.altitude_map_init))
        return table

    def assign_worker(self, worker_id, task_id):
        assert self.table[task_id, TC['accessible']] == 1
        # calculate dt
        worker_coords = self.get_coords(worker_id)
        task_coords = self.get_coords(task_id)
        task_process_time = self.table[task_id, TC['process_time']]

        distance = pdist(np.vstack([worker_coords, task_coords]), metric='cityblock')[0]  # calculate dt
        dt = distance + task_process_time
        self.table[worker_id, TC['dt']] = dt
        self.table[task_id, TC['dt']] = dt

        self.table[worker_id, TC['assigned']] = 1
        self.table[task_id, TC['assigned']] = 1

        self.table[worker_id, TC['target']] = task_id
        self.table[task_id, TC['target']] = worker_id

        print('w_id:', worker_id, '->', task_id, 'from:', worker_coords, ' to:', task_coords, ' dt:', dt)

        return dt

    def update(self):
        # event_trigger = False
        # busy_worker_coords = self.get_coords(busy_worker_ids)
        # target_coords = self.get_coords(target_ids)
        # busy_ids = np.union1d(busy_worker_ids, target_ids)

        busy_worker_ids = self.get_busy_worker_ids()
        busy_workers = self.table[busy_worker_ids, :]

        target_ids = busy_workers[:, TC['target']].astype(int)

        w2t_dt = busy_workers[:, TC['dt']]  # get worker to target distance
        dt_trigger = np.amin(w2t_dt)  # triggering event is the closest target

        done_workers = busy_workers[(busy_workers[:, TC['dt']] == dt_trigger), :]
        done_worker_ids = done_workers[:, TC['env_id']].astype(int)
        done_task_ids = done_workers[:, TC['target']].astype(int)

        for t in range(int(dt_trigger)):
            self.env_ticks += 1
            self.relocate_workers(busy_worker_ids, target_ids)
            self.trajectories.append((self.env_ticks, deepcopy(self.get_worker_coords())))  # add to trajectory

        for i in range(done_workers.shape[0]):
            self.complete_assignment(done_worker_ids[i], done_task_ids[i])

        self.demand_history.append((self.env_ticks, deepcopy(self.demand_map)))
        self.altitude_history.append((self.env_ticks, deepcopy(self.altitude_map)))

        done = False
        if len(self.get_active_cell_ids()) == 0:
            done = True

        return done, dt_trigger, done_worker_ids

    def get_coords(self, ids):
        ix_mask = (type(ids) is np.ndarray) and (len(ids.shape) > 0)
        if ix_mask:
            return self.table[np.ix_(ids, get_cols(self.coord_col_names, TC))]
        else:
            return self.table[ids, get_cols(self.coord_col_names, TC)]

    def construct_distance_matrices(self):
        worker_coords = self.get_worker_coords()
        cell_coords = self.get_active_cell_coords()
        self.cell_coords = cell_coords

        t2t = cdist(cell_coords, cell_coords, metric='cityblock')
        w2t = cdist(worker_coords, cell_coords, metric='cityblock')
        w2w = cdist(worker_coords, worker_coords, metric='cityblock')

        self.t2t = t2t  # type: np.array()
        self.w2t = w2t  # type: np.array()
        self.w2w = w2w  # type: np.array()

    def update_worker_dist(self):
        worker_coords = self.get_worker_coords()
        self.w2w = cdist(worker_coords, worker_coords, metric='cityblock')
        self.w2t = cdist(worker_coords, self.cell_coords, metric='cityblock')

    def get_worker_coords(self):
        return self.table[np.ix_(self.worker_ids, get_cols(self.coord_col_names, TC))]

    def get_active_cell_coords(self):
        mask = (self.table[self.cell_ids, TC['active']] == 1)
        return self.table[self.cell_ids][np.ix_(mask, get_cols(self.coord_col_names, TC))]

    def get_accessible_cell_ids(self):
        mask = (self.table[self.cell_ids, TC['accessible']] == 1)
        return self.table[self.cell_ids][mask, TC['env_id']].astype(int)

    def get_unassigned_worker_ids(self):
        mask = (self.table[self.worker_ids, TC['active']] == 1) & (self.table[self.worker_ids, TC['assigned']] == 0)
        return self.table[self.worker_ids][mask, TC['env_id']].astype(int)

    def get_busy_worker_ids(self):
        mask = (self.table[self.worker_ids, TC['active']] == 1) & (self.table[self.worker_ids, TC['assigned']] == 1)
        return self.table[self.worker_ids][mask, TC['env_id']].astype(int)

    def relocate_workers(self, busy_worker_ids, target_ids):
        busy_ids = np.union1d(busy_worker_ids, target_ids)
        self.table[busy_ids, TC['dt']] -= 1.0
        for i, worker_id in enumerate(busy_worker_ids):
            task_id = target_ids[i]
            worker_coords = self.get_coords(worker_id)
            task_coords = self.get_coords(task_id)

            if (task_coords == worker_coords).all():  # worker has arrived to task coords
                continue

            unit_step = self.get_unit_step_direction(task_coords, worker_coords)  # Direction for a unit step
            new_worker_coords = worker_coords + unit_step
            self.table[worker_id, get_cols(self.coord_col_names, TC)] = new_worker_coords
            # print('w_id:', worker_id, ' | ', 'task:', task_coords, ' | ', 'worker:', new_worker_coords, ' | ', 'tick:',
            #       self.env_ticks)

    def complete_assignment(self, worker_id, task_id):
        assert self.table[worker_id, TC['dt']] == 0.0
        assert self.table[task_id, TC['dt']] == 0.0

        self.table[worker_id, TC['dt']] = -1
        self.table[worker_id, TC['target']] = -1
        self.table[worker_id, TC['assigned']] = 0

        self.table[task_id, TC['active']] = 0
        self.table[task_id, TC['accessible']] = 0
        self.table[task_id, TC['dt']] = -1
        self.table[task_id, TC['target']] = -1
        self.table[task_id, TC['assigned']] = 0
        self.table[task_id, TC['visited_worker_id']] = worker_id

        if self.table[task_id, TC['has_ug']]:  # if deeper (underground) cells exist at the given x,y coords.
            self.table[task_id + 1, TC['accessible']] = 1  # mark underground cell as accessible.
            r, c = self.get_row_col(task_id)  # get row col coords of the task on altitude map.
            self.altitude_map[r, c] -= 1  # decrement altitude map at given coords.
            self.demand_map[r, c] -= 1  # decrement demand map at given coords.
        self.visited_cell_history[worker_id].append(task_id)

    def get_cells_at_coords(self, task_coords):
        mask = (self.table[self.cell_ids, TC['active']] == 1) & \
               (self.table[self.cell_ids, TC['x']] == task_coords[0]) & \
               (self.table[self.cell_ids, TC['y']] == task_coords[1]) & \
               (self.table[self.cell_ids, TC['z']] == task_coords[2])
        out = self.table[self.cell_ids][mask, TC['env_id']].astype(int)
        if len(out) > 0:
            return out
        else:
            return None

    @staticmethod
    def get_unit_step_direction(task_coords, worker_coords):
        delta = task_coords - worker_coords
        step = np.array([0, 0, 0])  # step in x, y, z directions
        for dim, val in enumerate(delta):
            if val == 0.0:
                continue
            else:
                step[dim] = np.sign(val)
                break
        return step

    def get_row_col(self, env_id):
        return self.table[env_id, get_cols(['row', 'col'], TC)].astype(int)

    def get_active_cell_ids(self):
        mask = self.table[self.cell_ids, TC['active']] == 1
        return self.table[self.cell_ids][mask, TC['env_id']].astype(int)

    def observe(self, worker_id):
        pass

    def get_available_task_ids(self):
        mask = (self.table[self.cell_ids, TC['active']] == 1) & (self.table[self.cell_ids, TC['assigned']] == 0) & (self.table[self.cell_ids, TC['accessible']] == 1)
        return self.table[self.cell_ids][mask, TC['env_id']].astype(int)

# if __name__ == '__main__':
#     env = Environment(env_mode='train')
#     for _ in range(10):
#         accessible_task_ids = env.get_accessible_cell_ids().tolist()
#         if len(accessible_task_ids) == 0:
#             break
#         env.assign_worker(env.worker_ids[0], accessible_task_ids.pop(0))
#         env.assign_worker(env.worker_ids[1], accessible_task_ids.pop())
#         done_g, dt_trigger_g, done_worker_ids_g = env.update()
#         print('--')
