from utils import *
from scipy.spatial.distance import cdist, pdist
from scipy.ndimage.filters import maximum_filter, minimum_filter
from itertools import product
import pandas as pd
from torch import from_numpy
import os
from sklearn.datasets.samples_generator import make_blobs


class Environment:
    def __init__(self,
                 env_mode: str,
                 test_data_dict: dict = None,
                 ):
        self.env_mode = env_mode
        self.env_ticks = 0
        self.test_map_dict = test_data_dict

        # INPUTS
        self.gen_method = None
        self.num_workers = None
        self.worker_coords_init = None
        self.demand_map_init = None
        self.altitude_map_init = None
        self.num_cells = None
        self.demand_map = None
        self.altitude_map = None
        self.nb_max, self.nb_min = None, None
        # -------------------------------------------------------------------------------------------------------------
        self.table, self.height, self.width, self.cell_ids, self.worker_ids, self.cell_coords = None, None, None, None, None, None
        self.t2t = None
        self.w2t = None
        self.w2w = None
        self.nb_max, self.nb_min = None, None
        self.entropy_matrix = None
        self.visited_cells_buffer = None
        self.worker_trace_buffer = []  # (env_ticks, worker_coords)
        self.demand_map_buffer = []  # (env_ticks, demand_map)
        self.altitude_map_buffer = []  # (env_ticks, altitude_map)
        self.coord_col_names = ['x', 'y', 'z']
        self.num_completed_cells = 0
        self.nb_fp = np.array([[1, 1, 1],  # 8-neighborhood
                               [1, 0, 1],
                               [1, 1, 1]])
        self.entropy_multiplier = 6
        self.initialize()

    def initialize(self):
        if self.env_mode == 'train':
            generation_methods = ['linear', 'random', 'hill']
            self.gen_method = gen_method = np.random.choice(generation_methods)
            # TODO: RANDOMIZE NUMBER OF WORKERS
            # TODO: RANDOMIZE ALTITUDE MAP
            # TODO: RANDOMIZE DEMAND MAP
            self.num_workers = 2
            max_depth = 3
            num_rows, num_cols = np.random.randint(3, 7, size=2)
            self.worker_coords_init = np.array([
                [0, 0],
                [num_cols - 1, num_rows - 1]
            ])
            if gen_method == 'random':
                self.demand_map_init = np.random.randint(0, max_depth + 1, size=[num_rows, num_cols])
            elif gen_method == 'linear':
                self.demand_map_init = self.generate_linear_soil(num_rows=num_rows,
                                                                 num_cols=num_cols,
                                                                 max_depth=max_depth)
            elif gen_method == 'hill':
                self.demand_map_init = self.generate_hill_soil(num_rows=num_rows,
                                                               num_cols=num_cols,
                                                               max_depth=max_depth)

            self.altitude_map_init = np.zeros_like(self.demand_map_init)
            self.table = self.construct_table()

        elif self.env_mode == 'test':
            self.altitude_map_init, self.demand_map_init, self.worker_coords_init = self.parse_test_data()
            self.num_workers = self.worker_coords_init.shape[0]
            self.table = self.construct_table()

        self.construct_distance_matrices()
        self.update_neighbourhood()

    def construct_table(self):
        self.num_cells = self.demand_map_init.sum()
        self.demand_map = deepcopy(self.demand_map_init)
        self.altitude_map = deepcopy(self.altitude_map_init)

        self.height, self.width = self.demand_map_init.shape
        self.table = table = -99 * np.ones([self.num_cells + self.num_workers, n_cols_table], dtype='float')
        self.cell_ids = np.arange(self.num_cells)
        self.worker_ids = np.arange(self.num_workers) + self.num_cells
        num_rows, num_cols = self.height, self.width

        # -------------------------------------------------------------------------------------------------------------
        # IMPUTE CELL COORDS
        table[:, TC['accessible']] = 0
        table[:, TC['has_ug']] = 0
        cell_id = 0
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

        # -------------------------------------------------------------------------------------------------------------
        # IMPUTE WORKER COORDS
        for w, worker_id in enumerate(self.worker_ids):
            x, y = self.worker_coords_init[w][0], self.worker_coords_init[w][1]
            table[worker_id, TC['env_id']] = worker_id
            z_row = num_rows - 1 - y
            z_col = x
            table[worker_id, get_cols(self.coord_col_names, TC)] = x, y, self.altitude_map_init[z_row, z_col]
        # -------------------------------------------------------------------------------------------------------------

        table[:, TC['active']] = 1
        table[:, TC['assigned']] = 0
        table[self.cell_ids, TC['soil']] = 1
        table[self.cell_ids, TC['worker']] = 0
        table[self.cell_ids, TC['process_time']] = 0

        table[self.worker_ids, TC['soil']] = 0
        table[self.worker_ids, TC['worker']] = 1
        table[self.worker_ids, TC['accessible']] = 1

        # ----------------------------------------MARK CELL ID OF TASK UNDER WORKER -----------------------------------
        worker_coords_init = self.get_coords(self.worker_ids, table)
        for i, w_coord in enumerate(worker_coords_init):
            w_id = self.worker_ids[i]
            under_cell_ids = self.get_cells_at_coords_xy(w_coord)
            if under_cell_ids is not None:
                self.table[w_id, TC['under_task_id']] = under_cell_ids[0]
        # -------------------------------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------------------------------
        # ENVIRONMENT EPISODE MEMORY BUFFER
        self.visited_cells_buffer = dict((i, []) for i in self.worker_ids)
        self.push_to_buffer(worker_coords_init, buffer=self.worker_trace_buffer, env_ticks=self.env_ticks)
        self.push_to_buffer(self.demand_map_init, buffer=self.demand_map_buffer, env_ticks=self.env_ticks)
        self.push_to_buffer(self.altitude_map_init, buffer=self.altitude_map_buffer, env_ticks=self.env_ticks)

        # self.worker_trace_buffer.append((self.env_ticks, worker_coords_init))
        # self.demand_map_buffer.append((self.env_ticks, self.demand_map_init))
        # self.altitude_map_buffer.append((self.env_ticks, self.altitude_map_init))

        return table

    @staticmethod
    def push_to_buffer(item, buffer, env_ticks):
        temp = np.array([deepcopy(env_ticks), deepcopy(item)])
        buffer.append(temp)

    def assign_worker(self, worker_id, task_id):
        assert self.table[task_id, TC['accessible']] == 1

        # calculate dt
        worker_coords = self.get_coords(worker_id)
        task_coords = self.get_coords(task_id)
        task_process_time = self.table[task_id, TC['process_time']]

        distance = pdist(np.vstack([worker_coords, task_coords]), metric='cityblock')[0]  # calculate dt

        r, c = self.table[task_id, TC['row']].astype(int), self.table[task_id, TC['col']].astype(int)
        entropy = self.entropy_matrix[r, c]

        if entropy >= 0:
            mult = 0.5
        else:
            mult = self.entropy_multiplier

        self.table[worker_id, TC['dt']] = distance
        self.table[task_id, TC['dt']] = distance

        self.table[worker_id, TC['assigned']] = 1
        self.table[task_id, TC['assigned']] = 1

        self.table[worker_id, TC['target']] = task_id
        self.table[task_id, TC['target']] = worker_id
        self.table[worker_id, TC['under_task_id']] = -1
        reward = -distance - task_process_time + (entropy * mult)
        return reward

    def update(self):
        # event_trigger = False
        # busy_worker_coords = self.get_coords(busy_worker_ids)
        # target_coords = self.get_coords(target_ids)
        # busy_ids = np.union1d(busy_worker_ids, target_ids)

        busy_worker_ids = self.get_busy_worker_ids()
        busy_workers = self.table[busy_worker_ids, :]
        if len(busy_worker_ids) == 0:
            pass
        target_ids = busy_workers[:, TC['target']].astype(int)

        w2t_dt = busy_workers[:, TC['dt']]  # get worker to target distance
        dt_trigger = np.amin(w2t_dt)  # triggering event is the closest target

        done_workers = busy_workers[(busy_workers[:, TC['dt']] == dt_trigger), :]
        done_worker_ids = done_workers[:, TC['env_id']].astype(int)
        done_task_ids = done_workers[:, TC['target']].astype(int)

        self.relocate_workers(busy_worker_ids, target_ids, dt_trigger)

        for i in range(done_workers.shape[0]):
            self.complete_assignment(done_worker_ids[i], done_task_ids[i])

        self.update_worker_dist()
        self.update_neighbourhood()

        self.push_to_buffer(self.demand_map, buffer=self.demand_map_buffer, env_ticks=self.env_ticks)
        self.push_to_buffer(self.altitude_map, buffer=self.altitude_map_buffer, env_ticks=self.env_ticks)

        done = False
        if len(self.get_active_cell_ids()) == 0:
            done = True

        return done, dt_trigger, done_worker_ids

    def update_neighbourhood(self):
        self.nb_max = maximum_filter(self.altitude_map, footprint=self.nb_fp, mode='mirror')
        self.nb_min = minimum_filter(self.altitude_map, footprint=self.nb_fp, mode='mirror')
        self.entropy_matrix = (self.altitude_map - self.nb_min)
        accessible_cell_ids = self.get_accessible_cell_ids()
        for cell_id in accessible_cell_ids:
            r, c = self.table[cell_id, TC['row']].astype(int), self.table[cell_id, TC['col']].astype(int)
            nb_min, nb_max = self.nb_min[r, c], self.nb_max[r, c]
            self.table[cell_id, TC['nb_min']] = nb_min
            self.table[cell_id, TC['nb_max']] = nb_max

    def get_coords(self, ids, table=None):
        if table is None:
            table = self.table
        ix_mask = (type(ids) is np.ndarray) and (len(ids.shape) > 0)
        if ix_mask:
            return table[np.ix_(ids, get_cols(self.coord_col_names, TC))]
        else:
            return table[ids, get_cols(self.coord_col_names, TC)]

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

    def get_inaccessible_cell_ids(self):
        mask = (self.table[self.cell_ids, TC['accessible']] == 0) & (self.table[self.cell_ids, TC['active']] == 1)
        return self.table[self.cell_ids][mask, TC['env_id']].astype(int)

    def get_unassigned_worker_ids(self):
        mask = (self.table[self.worker_ids, TC['active']] == 1) & (self.table[self.worker_ids, TC['assigned']] == 0)
        return self.table[self.worker_ids][mask, TC['env_id']].astype(int)

    def get_busy_worker_ids(self):
        mask = (self.table[self.worker_ids, TC['active']] == 1) & (self.table[self.worker_ids, TC['assigned']] == 1)
        return self.table[self.worker_ids][mask, TC['env_id']].astype(int)

    def move_workers_unit_step(self, busy_worker_ids, target_ids):
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

        self.table[worker_id, TC['under_task_id']] = -1
        if self.table[task_id, TC['has_ug']]:  # if deeper (underground) cells exist at the given x,y coords.
            self.table[task_id + 1, TC['accessible']] = 1  # mark underground cell as accessible.
            self.table[worker_id, TC['under_task_id']] = task_id + 1

        r, c = self.get_row_col(task_id)  # get row col coords of the task on altitude map.
        self.altitude_map[r, c] -= 1  # decrement altitude map at given coords.
        self.demand_map[r, c] -= 1  # decrement demand map at given coords.
        self.visited_cells_buffer[worker_id].append((self.env_ticks, task_id))

    def get_cells_at_coords_xy(self, task_coords):
        mask = (self.table[self.cell_ids, TC['active']] == 1) & \
               (self.table[self.cell_ids, TC['x']] == task_coords[0]) & \
               (self.table[self.cell_ids, TC['y']] == task_coords[1])
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

    def observe(self,
                target_worker_id: int,
                fix_num_nodes: bool = True):
        # define structure of g-edges and g-nodes arrays
        n_types = ['task', 'worker']
        n_status = ['accessible', 'inaccessible', 'assigned', 'unassigned', 'has_ug', 'not_has_ug']
        n_features = ['dt', 'nb_min', 'nb_max']

        e_types = ['t2t', 't2w', 'w2t', 'w2w']
        e_status = []
        e_features = ['dist']

        g_node_col_names = ['env_id', 'g_id'] + n_types + n_status + n_features
        g_edge_col_names = ['e_id', 'env_src', 'env_dst', 'g_src', 'g_dst',
                            'action_space'] + e_features + e_types + e_status

        n_cols_edge = len(g_edge_col_names)
        ec = dict((g_edge_col_names[i], i) for i in range(n_cols_edge))

        n_cols_node = len(g_node_col_names)
        nc = dict((g_node_col_names[i], i) for i in range(n_cols_node))

        active_cell_ids = self.get_active_cell_ids()

        n_nodes = len(active_cell_ids) + len(self.worker_ids)  # total number of active (existing) nodes in graph
        env_node_ids = np.concatenate([active_cell_ids, self.worker_ids])
        g_node_ids = np.arange(n_nodes)

        # map env id to graph id
        e2g_dict = dict((env_id, g_id) for env_id, g_id in zip(env_node_ids, g_node_ids))

        def e2g_map(x):
            return e2g_dict[int(x)]

        e2g_vec = np.vectorize(e2g_map)  # vectorize e2g_map()

        # ----------------------- CONSTRUCT EDGES --------------------------
        accessible_cell_ids = self.get_accessible_cell_ids()

        # accessible 2 accessible (fully connected)
        acc2acc_src_dst = np.array(list(product(accessible_cell_ids, accessible_cell_ids)))
        acc2acc_edges = np.zeros([acc2acc_src_dst.shape[0], n_cols_edge])
        acc2acc_edges[:, get_cols(['env_src', 'env_dst'], ec)] = acc2acc_src_dst
        acc2acc_edges[:, ec['t2t']] = 1
        acc2acc_edges[:, ec['dist']] = self.t2t[np.ix_(accessible_cell_ids, accessible_cell_ids)].reshape(-1)

        # surface 2 underground, ug2ug-beneath,
        acc2ug_edges = []
        for surface_id in accessible_cell_ids:
            if not self.table[surface_id, TC['has_ug']]:
                continue
            else:
                top_id = surface_id
                d = 1
                while self.table[top_id, TC['has_ug']]:
                    edge_1 = np.zeros(n_cols_edge)
                    edge_2 = np.zeros(n_cols_edge)
                    edge_3 = np.zeros(n_cols_edge)
                    edge_4 = np.zeros(n_cols_edge)
                    bot_id = top_id + 1

                    edge_1[get_cols(['env_src', 'env_dst', 't2t', 'dist'],
                                    ec)] = top_id, bot_id, 1, 1  # distance between top-bot is 1
                    edge_2[get_cols(['env_src', 'env_dst', 't2t', 'dist'], ec)] = bot_id, top_id, 1, 1

                    acc2ug_edges.append(edge_1)
                    acc2ug_edges.append(edge_2)

                    if top_id != surface_id:  # connect underground nodes with surface
                        edge_3[get_cols(['env_src', 'env_dst', 't2t', 'dist'], ec)] = surface_id, bot_id, 1, d
                        edge_4[get_cols(['env_src', 'env_dst', 't2t', 'dist'], ec)] = bot_id, surface_id, 1, d
                        acc2ug_edges.append(edge_3)
                        acc2ug_edges.append(edge_4)

                    d += 1
                    top_id += 1
        acc2ug_edges = np.array(acc2ug_edges).reshape(-1, n_cols_edge)

        # worker2task edges
        w2t_src_dst = np.array(list(product(self.worker_ids, accessible_cell_ids)))
        w2t_edges = np.zeros([w2t_src_dst.shape[0], n_cols_edge])
        w2t_edges[:, get_cols(['env_src', 'env_dst'], ec)] = w2t_src_dst
        w2t_edges[:, ec['w2t']] = 1
        w2t_edges[:, ec['dist']] = self.w2t[np.ix_(self.worker_ids - self.num_cells, accessible_cell_ids)].reshape(-1)

        t2w_edges = deepcopy(w2t_edges)
        t2w_edges[:, ec['env_src']] = w2t_edges[:, ec['env_dst']]
        t2w_edges[:, ec['env_dst']] = w2t_edges[:, ec['env_src']]
        t2w_edges[:, ec['w2t']] = 0
        t2w_edges[:, ec['t2w']] = 1

        w2w_src_dst = np.array(list(product(self.worker_ids, self.worker_ids)))
        w2w_edges = np.zeros([w2w_src_dst.shape[0], n_cols_edge])
        w2w_edges[:, get_cols(['env_src', 'env_dst'], ec)] = w2w_src_dst
        w2w_edges[:, ec['w2w']] = 1
        w2w_edges[:, ec['dist']] = self.w2w[
            np.ix_(self.worker_ids - self.num_cells, self.worker_ids - self.num_cells)].reshape(-1)

        # acc2acc_edges[:, ec['inaccessible']] = acc2acc_edges[:, ec['accessible']] == 0
        g_edges = np.vstack([acc2acc_edges, acc2ug_edges, w2t_edges, t2w_edges, w2w_edges])
        g_edges[:, ec['g_src']] = e2g_vec(g_edges[:, ec['env_src']])
        g_edges[:, ec['g_dst']] = e2g_vec(g_edges[:, ec['env_dst']])
        g_edges[:, ec['e_id']] = np.arange(g_edges.shape[0])
        zero_dist_mask = g_edges[:, ec['dist']] == 0
        g_edges[zero_dist_mask, ec['dist']] += VERY_SMALL_NUMBER  # avoiding zero valued distances

        g_nodes = np.zeros([n_nodes, n_cols_node])

        tc_type_names = ['soil', 'worker']  # tc == table column
        g_type_names = ['task', 'worker']

        g_nodes[:, nc['env_id']] = env_node_ids
        g_nodes[:, nc['g_id']] = g_node_ids

        g_nodes[:, get_cols(g_type_names, nc)] = self.table[np.ix_(env_node_ids, get_cols(tc_type_names, TC))]
        g_nodes[:, get_cols(['accessible', 'assigned', 'has_ug'], nc)] = self.table[
            np.ix_(env_node_ids, get_cols(['accessible', 'assigned', 'has_ug'], TC))]
        g_nodes[:, get_cols(['dt', 'nb_min', 'nb_max'], nc)] = self.table[np.ix_(env_node_ids, get_cols(['dt'], TC))]
        g_nodes[:, nc['unassigned']] = g_nodes[:, nc['assigned']] == 0
        g_nodes[:, nc['inaccessible']] = g_nodes[:, nc['accessible']] == 0
        g_nodes[:, nc['not_has_ug']] = g_nodes[:, nc['has_ug']] == 0
        g_nodes[g_nodes == -99] = 0  # mask -1 values (aka N/A) to 0

        available_task_ids = self.get_available_task_ids(target_worker_id)

        src_mask = np.in1d(g_edges[:, ec['env_src']], target_worker_id)  # where source is target worker
        dst_mask = np.in1d(g_edges[:, ec['env_dst']], available_task_ids)  # where destination is target op
        action_space_mask = src_mask & dst_mask
        g_edges[:, ec['action_space']] = action_space_mask

        def _to_categorical(data_arr, col_names, col_map):
            return pd.DataFrame(data_arr[:, get_cols(col_names, col_map)]).idxmax(axis=1).to_numpy()

        n_type_arr = from_numpy(_to_categorical(g_nodes, n_types, nc)).float().to(DEVICE)
        e_type_arr = from_numpy(_to_categorical(g_edges, e_types, ec)).float().to(DEVICE)

        # DGL GRAPH
        g = dgl.DGLGraph()
        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)

        g_node_features = from_numpy(g_nodes[:, get_cols(n_status + n_features, nc)]).float().to(DEVICE)
        g_edge_features = from_numpy(g_edges[:, get_cols(e_status + e_features, ec)]).float().to(DEVICE)

        node_dict = {'nf_init': g_node_features,
                     'n_type': n_type_arr}

        edge_dict = {
            'ef_init': g_edge_features,
            'action_space': from_numpy(g_edges[:, ec['action_space']]).float().to(DEVICE),
            'e_type': e_type_arr,
            'g2e': from_numpy(g_edges[:, ec['env_dst']]).float().to(DEVICE),
        }

        g.add_nodes(n_nodes, node_dict)
        g.add_edges(g_edges[:, ec['g_src']], g_edges[:, ec['g_dst']], edge_dict)

        action_space_cardinality = g_edges[:, ec['action_space']].sum()  # number of potential actions
        is_trivial = False
        action_space_env_ids = g_edges[action_space_mask, ec['env_dst']].astype(int)

        if action_space_cardinality <= 1:
            is_trivial = True

        return g, is_trivial, action_space_env_ids

    def get_available_task_ids(self, target_worker_id):
        accessible_active_unassigned_mask = (self.table[self.cell_ids, TC['active']] == 1) & (
                self.table[self.cell_ids, TC['assigned']] == 0) & (
                                                    self.table[self.cell_ids, TC['accessible']] == 1)

        other_unassigned_worker_ids = np.setdiff1d(self.get_unassigned_worker_ids(), target_worker_id)
        under_task_ids = self.table[other_unassigned_worker_ids, TC['under_task_id']].astype(int)
        under_task_ids = np.setdiff1d(under_task_ids, -1)
        under_mask = ~np.in1d(self.cell_ids, under_task_ids)
        mask = accessible_active_unassigned_mask & under_mask
        return self.table[self.cell_ids][mask, TC['env_id']].astype(int)

    def relocate_workers(self, busy_worker_ids, target_ids, dt_trigger):
        for t in range(int(dt_trigger)):
            self.env_ticks += 1
            self.move_workers_unit_step(busy_worker_ids, target_ids)
            self.push_to_buffer(self.get_worker_coords(), buffer=self.worker_trace_buffer, env_ticks=self.env_ticks)

    def export(self):
        """
        visited_cell_history
        demand_history
        trajectories
        altitude_history
        :return:
        """
        env_snapshots = np.array(
            [self.demand_map_buffer,
             self.altitude_map_buffer,
             self.worker_trace_buffer,
             self.visited_cells_buffer]
        )
        return env_snapshots

    @staticmethod
    def generate_linear_soil(num_rows, num_cols, max_depth):

        demand_map = np.zeros((num_rows, num_cols), dtype="int32")
        demand_map[0, :] = np.random.randint(1, max_depth)  # fill the first row with random number

        for r in range(1, num_rows):
            d = np.random.randint(1, max_depth)
            demand_map[r, :] = d

        return demand_map

    @staticmethod
    def generate_hill_soil(num_rows: int,
                           num_cols: int,
                           max_depth: int,
                           ):
        n_hills = np.random.randint(1, 3)
        x, truth = make_blobs(n_samples=500, centers=n_hills, cluster_std=np.random.uniform(2, 3, n_hills))
        hist = np.histogram2d(x[:, 0], x[:, 1], bins=[num_rows, num_cols])
        a = hist[0]

        demand_map = np.round(np.interp(a, (a.min(), a.max()), (0, max_depth)))
        demand_map = demand_map.astype(int)
        return demand_map

    # noinspection DuplicatedCode
    def parse_test_data(self):
        altitude_map = self.test_map_dict['altitude_map']
        num_rows, num_cols = altitude_map.shape
        worker_rows, worker_cols = np.where(altitude_map == 90)
        worker_ys = (num_rows - 1 - worker_rows).reshape(-1, 1)
        worker_xs = worker_cols.reshape(-1, 1)
        worker_coords = np.hstack([worker_xs, worker_ys])
        altitude_map[altitude_map == 90] = 0

        demand_map = altitude_map - self.test_map_dict['required_map']

        return altitude_map, demand_map, worker_coords
