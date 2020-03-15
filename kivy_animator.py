from kivy_renderer.kivy_utils import *
from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder
import numpy as np
import os
import moviepy.editor as mpy
import datetime
import time

MAP_HEIGHT = 300

class EnvHandler:
    def __init__(self, env_data_path):
        self.demand_map_buffer, self.altitude_map_buffer, self.worker_trace_buffer, self.visited_cells_buffer = np.load(
            env_data_path, allow_pickle=True)
        self.demand_map_init = deepcopy(self.demand_map_buffer[0][1])
        self.altitude_map_init = deepcopy(self.altitude_map_buffer[0][1])
        self.worker_coords_init = deepcopy(self.worker_trace_buffer[0][1])

        self.demand_map_buffer = np.concatenate(self.demand_map_buffer).reshape(-1, 2)
        self.altitude_map_buffer = np.concatenate(self.altitude_map_buffer).reshape(-1, 2)
        self.worker_trace_buffer = np.concatenate(self.worker_trace_buffer).reshape(-1, 2)
        self.env_ticks_max = self.worker_trace_buffer[:, 0].max()
        self.tick_snapshots = self.create_tick_snapshots()
        self.num_rows, self.num_cols = self.demand_map_buffer[0, 1].shape
        self.num_workers = self.worker_trace_buffer[0, 1].shape[0]
        # self.altitude_map_buffer[0, 1][0, 0] = 3

        max_altitude, min_altitude = self.altitude_map_buffer[0, 1].max(), self.altitude_map_buffer[-1, 1].min()
        self.altitude_range = altitude_range = np.arange(min_altitude, max_altitude + 1).astype(int)
        colors_list = self.get_colors_list(altitude_range)
        self.cell_color_map = dict((altitude, c) for altitude, c in zip(altitude_range, colors_list))
        self.num_completed_cells = 0
        self.num_total_demanded_volume = self.demand_map_init.sum()

    def update_completed_cells(self, tick_id):
        demand_map_current = self.tick_snapshots[tick_id]['demand_map']  # type: np.array
        if demand_map_current is not None:
            self.num_completed_cells = self.num_total_demanded_volume - demand_map_current.sum()

    def create_tick_snapshots(self):
        tick_snapshots = []

        for tick in range(self.env_ticks_max + 1):
            tick_snapshots.append(
                dict(env_tick=tick,
                     demand_map=None,
                     altitude_map=None,
                     worker_coords=None)
            )
        for t, item in self.demand_map_buffer:
            tick_snapshots[t]['demand_map'] = item
        for t, item in self.altitude_map_buffer:
            tick_snapshots[t]['altitude_map'] = item
        for t, item in self.worker_trace_buffer:
            tick_snapshots[t]['worker_coords'] = item
        return tick_snapshots

    @staticmethod
    def get_colors_list(altitude_range):
        above_color_from = clr.Color('#be79df')
        above_color_to = clr.Color('#21243d')  # light yellow

        below_color_from = clr.Color('#4c3100')  # dark red
        below_color_to = clr.Color('#ffa500')  # light orange

        above_ground_levels = altitude_range[altitude_range > 0]
        under_ground_levels = altitude_range[altitude_range < 0]

        colors_above = []
        if len(above_ground_levels) > 0:
            colors_range = list(above_color_from.range_to(above_color_to, len(above_ground_levels)))
            colors_above = [c.get_rgb() for c in colors_range]

        colors_below = []
        if len(under_ground_levels) > 0:
            colors_range = list(below_color_from.range_to(below_color_to, len(under_ground_levels)))
            colors_below = [c.get_rgb() for c in colors_range]

        color_zero = [clr.Color('#ececec').get_rgb()]

        colors_list = colors_below + color_zero + colors_above
        return colors_list


class Cell(Widget):
    handler = ObjectProperty(None)  # type: EnvHandler
    hidden = NumericProperty(0)
    cell_altitude = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Cell, self).__init__(**kwargs)

        cell_color = self.handler.cell_color_map[int(self.cell_altitude)]
        with self.canvas.before:
            self.fill_color = Color(rgb=cell_color)
            self.rectangle = Rectangle(size=self.size, pos=self.pos)

        with self.canvas:
            self.border_color = Color(rgba=CONFIG['cell_border_color'])
            self.border = Line(
                width=CONFIG['cell_border_width'],
                rectangle=[self.x, self.y, self.width, self.height],
            )
        self.bind(cell_altitude=self.update_cell_color)

    def update_cell_color(self, *args):
        cell_color = self.handler.cell_color_map[int(self.cell_altitude)]
        self.fill_color.rgb = cell_color


class Worker(Widget):
    worker_id = NumericProperty(None)
    handler = ObjectProperty(None)  # type: EnvHandler
    hidden = NumericProperty(0)

    def __init__(self, **kwargs):
        super(Worker, self).__init__(**kwargs)

        # worker_color = self.handler.worker_color_map[int(self.worker_id)]
        # worker_color = self.handler.worker_color_map[int(self.worker_id)]
        with self.canvas.before:
            self.fill_color = Color(rgba=get_rgba_color(WORKER_COLORS[self.worker_id]))
            self.circle = Ellipse(size=self.size, pos=self.pos)

        self.bind(center=self.update_center)

    def update_center(self, *args):
        self.circle.pos = self.pos


class VolumeColorBarWidget(Widget):
    handler = ObjectProperty(None)  # type: EnvHandler
    hidden = NumericProperty(0)

    def __init__(self, **kwargs):
        super(VolumeColorBarWidget, self).__init__(**kwargs)

        panel = PanelWidget(size=self.size,
                            pos=self.pos,
                            bg_color=get_rgba_color('black', a=1),
                            with_shadow=True,
                            elevation=3)
        self.add_widget(panel)

        num_cells = len(self.handler.altitude_range)
        cell_width = self.width
        cell_height = self.height / num_cells
        rect_y = self.y

        interval_label_pos_x = self.x + 1.3 * cell_width
        cell_size = cell_width, cell_height
        interval_label_size = cell_width, cell_height
        interval_label_pos_y = self.y

        for i, altitude in enumerate(self.handler.altitude_range):
            colored_rectangle = ColoredRectangle(
                color=self.handler.cell_color_map[altitude],
                x=self.x,
                y=rect_y,
                size=cell_size,
            )

            self.add_widget(colored_rectangle)

            label_kwargs = copy(CONFIG['label_kwargs'])
            label_kwargs['text'] = str(altitude)
            label_kwargs['halign'] = 'left'
            itv_label = LabelWidget(pos=[interval_label_pos_x, interval_label_pos_y],
                                    size=interval_label_size,
                                    label_kwargs=label_kwargs)
            self.add_widget(itv_label)
            rect_y += cell_height
            interval_label_pos_y += cell_height


class AltitudeMap(Widget):
    handler = ObjectProperty(None)  # type: EnvHandler
    hidden = NumericProperty(0)
    color = ListProperty(None)
    is_final = props.BooleanProperty(False)

    def __init__(self, **kwargs):
        super(AltitudeMap, self).__init__(**kwargs)

        with self.canvas.before:
            Color(rgba=CONFIG['map_bg_color'])
            Rectangle(pos=self.pos, size=self.size)

        panel = PanelWidget(size=self.size,
                            pos=self.pos,
                            bg_color=get_rgba_color('black', a=1),
                            with_shadow=True,
                            elevation=3)

        self.add_widget(panel)
        self.cells_map = None
        self.cell_size = None
        self.workers = []
        self.num_rows, self.num_cols = None, None
        self.num_workers = None
        if not self.is_final:
            self.initialize_altitude_map()
        elif self.is_final:
            self.initialize_demand_map()

    def initialize_altitude_map(self):
        altitude_map_init = self.handler.altitude_map_buffer[0][1]
        worker_coords_init = self.handler.worker_trace_buffer[0][1]
        self.num_workers = self.handler.num_workers

        self.num_rows, self.num_cols = altitude_map_init.shape
        max_dim = max(self.num_rows, self.num_cols)

        cell_height = self.height / max_dim
        self.cell_size = cell_size = cell_height, cell_height
        self.cells_map = np.zeros(shape=altitude_map_init.shape, dtype='object')

        # CREATE ALTITUDE MAP CELLS
        for r in np.arange(self.num_rows):
            for c in np.arange(self.num_cols):
                cell_pos = int(self.x + cell_height * c), int(self.y + self.height - cell_height * (r + 1))
                cell_altitude = altitude_map_init[r, c].astype(int)
                cell = Cell(handler=self.handler,
                            pos=cell_pos,
                            size=cell_size,
                            cell_altitude=cell_altitude,
                            )
                self.add_widget(cell)
                self.cells_map[r, c] = cell

        for w_id, worker_coords in enumerate(worker_coords_init):
            worker_center = self.get_worker_center(worker_coords)
            worker = Worker(handler=self.handler,
                            size=(cell_size[0] * 0.75, cell_size[0] * 0.75),
                            center=worker_center,
                            worker_id=w_id,
                            )
            self.add_widget(worker)
            self.workers.append(worker)

    def initialize_demand_map(self):
        self.num_workers = self.handler.num_workers
        # demand_map = -1 * self.handler.demand_map_buffer[0][1]
        # demand_map = self.handler.demand_map_buffer[0][1]
        demand_map = self.handler.tick_snapshots[-1]['altitude_map']

        self.num_rows, self.num_cols = demand_map.shape
        max_dim = max(self.num_rows, self.num_cols)

        cell_height = self.height / max_dim
        self.cell_size = cell_size = cell_height, cell_height
        self.cells_map = np.zeros(shape=demand_map.shape, dtype='object')

        for r in np.arange(self.num_rows):
            for c in np.arange(self.num_cols):
                cell_pos = int(self.x + cell_height * c), int(self.y + self.height - cell_height * (r + 1))
                cell_altitude = demand_map[r, c].astype(int)
                cell = Cell(handler=self.handler,
                            pos=cell_pos,
                            size=cell_size,
                            cell_altitude=cell_altitude,
                            )
                self.add_widget(cell)
                self.cells_map[r, c] = cell

    def update_map(self, tick_id):
        altitude_map = self.handler.tick_snapshots[tick_id]['altitude_map']
        worker_coords = self.handler.tick_snapshots[tick_id]['worker_coords']

        if altitude_map is not None:
            for r in np.arange(self.num_rows):
                for c in np.arange(self.num_cols):
                    new_altitude = altitude_map[r, c]
                    cell = self.cells_map[r, c]
                    cell.cell_altitude = new_altitude

        for w_id, worker_coord in enumerate(worker_coords):
            worker = self.workers[w_id]
            new_center = self.get_worker_center(worker_coords[w_id])
            worker.center = new_center

    def get_worker_center(self, worker_coords):
        w_x = int(self.x + worker_coords[0] * self.cell_size[0] + self.cell_size[0] / 2)
        w_y = int(self.y + worker_coords[1] * self.cell_size[1] + self.cell_size[1] / 2)
        return w_x, w_y


class MainUI(Widget):
    handler = ObjectProperty(None)  # type: EnvHandler
    hidden = NumericProperty(0)
    color = ListProperty(None)
    to_export = props.BooleanProperty(False)

    def __init__(self, **kwargs):
        super(MainUI, self).__init__(**kwargs)

        self.images = []
        self.env_ticks = 0
        with self.canvas.before:
            Color(rgb=self.color)
            Rectangle(pos=self.pos, size=self.size)

        # ------------------------------------------------------------------------------------------------------------
        # ALTITUDE MAP
        num_rows, num_cols = self.handler.num_rows, self.handler.num_cols
        max_dim = np.where(np.amax([num_rows, num_cols]))[0][0]
        dims = num_rows, num_cols
        height = 130
        altitude_map_pos = (100, height)

        k = MAP_HEIGHT
        # if max(self.handler.num_rows, self.handler.num_cols) < 7:
        #     k = 250
        altitude_map_size = [None, None]
        altitude_map_size[1 - max_dim] = k
        altitude_map_size[max_dim] = int(k * (dims[1 - max_dim] / dims[max_dim]))

        self.altitude_map = AltitudeMap(pos=altitude_map_pos, size=altitude_map_size, handler=self.handler)

        # ------------------------------------------------------------------------------------------------------------
        # ALTITUDE HEADER
        alt_text_pos = self.altitude_map.x, self.altitude_map.top
        alt_text_size = self.altitude_map.width, 40
        label_kwargs = copy(CONFIG['label_kwargs'])
        label_kwargs['text'] = 'Altitude Map'
        label_kwargs['halign'] = 'center'
        label_kwargs['valign'] = 'middle'
        label_kwargs['font_size'] = sp(25)
        label_kwargs['color'] = get_rgba_color('black', a=0.8)

        self.alt_label = LabelWidget(pos=alt_text_pos,
                                     size=alt_text_size,
                                     label_kwargs=label_kwargs
                                     )

        self.add_widget(self.altitude_map)
        self.add_widget(self.alt_label)

        # ------------------------------------------------------------------------------------------------------------
        # ENV TIME HEADER
        time_label_pos = self.altitude_map.x, 70
        time_text_size = self.altitude_map.width / 2, 30
        time_label_kwargs = copy(CONFIG['label_kwargs'])
        time_label_kwargs['halign'] = 'left'
        time_label_kwargs['valign'] = 'middle'
        time_label_kwargs['text'] = 'None'
        self.time_label = LabelWidget(pos=time_label_pos,
                                      size=time_text_size,
                                      label_kwargs=time_label_kwargs)
        self.add_widget(self.time_label)
        # ------------------------------------------------------------------------------------------------------------
        # NUM CELLS COMPLETED HEADER
        num_cells_completed_pos = self.altitude_map.x, 50
        num_cells_completed_size = self.altitude_map.width / 2, 30
        num_cells_label_kwargs = copy(time_label_kwargs)
        num_cells_label_kwargs['halign'] = 'left'
        num_cells_label_kwargs['text'] = 'None'

        self.num_cells_label = LabelWidget(pos=num_cells_completed_pos,
                                           size=num_cells_completed_size,
                                           label_kwargs=num_cells_label_kwargs)
        self.add_widget(self.num_cells_label)

        # ------------------------------------------------------------------------------------------------------------
        # TICK DEBUG BUTTON
        btn_size = 30, 30
        btn_pos = self.right - 100, height
        self.tick_btn = CustomButton(pos=btn_pos, size=btn_size, text='T')
        self.add_widget(self.tick_btn)
        # self.tick_btn.bind(on_press=self.write_video)

        # ------------------------------------------------------------------------------------------------------------
        # COLORBAR
        colorbar_pos = (altitude_map_pos[0] + altitude_map_size[0] + 50, height)
        colorbar_size = (30, 140)

        self.colorbar_widget = VolumeColorBarWidget(size=colorbar_size,
                                                    pos=colorbar_pos,
                                                    handler=self.handler)
        self.add_widget(self.colorbar_widget)

        # ALTITUDE HEADER
        colorbar_text_pos = self.colorbar_widget.x, self.colorbar_widget.top
        colorbar_text_size = 100, 40
        colorbar_text_kwargs = copy(CONFIG['label_kwargs'])
        colorbar_text_kwargs['text'] = 'Ground Level'
        colorbar_text_kwargs['halign'] = 'left'
        colorbar_text_kwargs['valign'] = 'middle'
        colorbar_text_kwargs['font_size'] = sp(15)
        colorbar_text_kwargs['color'] = get_rgba_color('black', a=0.8)

        self.colorbar_label = LabelWidget(pos=colorbar_text_pos,
                                          size=colorbar_text_size,
                                          label_kwargs=colorbar_text_kwargs
                                          )
        self.add_widget(self.colorbar_label)

        demand_map_pos = self.colorbar_widget.right + 100, height
        self.demand_map = AltitudeMap(pos=demand_map_pos,
                                      size=altitude_map_size,
                                      handler=self.handler,
                                      is_final=True,
                                      )
        self.add_widget(self.demand_map)

        # DEMAND HEADER
        demand_map_pos = self.demand_map.x, self.demand_map.top
        demand_map_size = self.demand_map.width, 40
        demand_map_kwargs = copy(CONFIG['label_kwargs'])
        demand_map_kwargs['text'] = 'Required map'
        demand_map_kwargs['halign'] = 'center'
        demand_map_kwargs['valign'] = 'middle'
        demand_map_kwargs['font_size'] = sp(25)
        demand_map_kwargs['color'] = get_rgba_color('black', a=0.8)

        self.demand_label = LabelWidget(pos=demand_map_pos,
                                        size=demand_map_size,
                                        label_kwargs=demand_map_kwargs
                                        )
        self.add_widget(self.demand_label)
        self.update_info_labels()
        if self.to_export:
            self._export_frame()

    def update(self, *args):

        self.env_ticks += 1
        # print('env ticks: ', self.env_ticks)
        if self.env_ticks < len(self.handler.tick_snapshots):
            self.altitude_map.update_map(self.env_ticks)
            self.handler.update_completed_cells(self.env_ticks)
            self.update_info_labels()
        else:
            hide_widget(self.tick_btn)

            # Clock.unschedule(self.update)

    def update_info_labels(self, *args):
        self.time_label.label.text = 'Time: ' + str(self.env_ticks)
        self.num_cells_label.label.text = 'Completed: ' + str(self.handler.num_completed_cells)

    def write_video(self, obj, movie_duration_max=30.0):
        assert self.to_export is True

        total_ticks = len(self.handler.tick_snapshots)
        interval = min(movie_duration_max / total_ticks, 0.5)
        for i in range(total_ticks):
            self.update()
            if self.to_export:
                self._export_frame()

        start_duration_sec = 2
        end_duration_sec = 2
        fps = int(1/interval)
        start_list = [self.images[0] for _ in range(int(start_duration_sec * fps))]
        end_list = [self.images[-1] for _ in range(int(end_duration_sec * fps))]
        images = start_list + self.images + end_list

        clip = mpy.ImageSequenceClip(images, fps=fps, with_mask=True)
        now = datetime.datetime.now()
        filename = now.strftime("%m%d_%H%M") + "_" + "animation.mp4"
        clip.write_videofile(os.path.join(filename), fps=fps)
        exit()

    def _export_frame(self):
        img = self.export_as_image()
        img_numpy = np.frombuffer(img.texture.pixels, dtype=np.ubyte).reshape(self.height, self.width, 4)
        self.images.append(img_numpy)

    def animate(self, obj, movie_duration_max: int = 30):
        assert self.to_export is False
        total_ticks = len(self.handler.tick_snapshots)
        interval = min(movie_duration_max / total_ticks, 0.2)
        Clock.schedule_interval(self.update, interval)


class SCKivyApp(App):
    def __init__(self,
                 data_path: os.path,
                 to_export: bool = False,
                 to_animate: bool = False,
                 **kwargs):
        super(SCKivyApp, self).__init__(**kwargs)
        self.main_layout = None
        self.to_export = to_export
        self.to_animate = to_animate
        self.handler = EnvHandler(data_path)

    def build(self):
        self.main_layout = MainUI(handler=self.handler,
                                  size_hint=(None, None),
                                  size=Window.size,
                                  color=CONFIG['main_bg_color'],
                                  to_export=self.to_export,
                                  )
        if self.to_export:
            self.main_layout.tick_btn.bind(on_press=self.main_layout.write_video)
        elif self.to_animate:
            self.main_layout.tick_btn.bind(on_press=self.main_layout.animate)
        else:
            self.main_layout.tick_btn.bind(on_press=self.main_layout.update)
        return self.main_layout


if __name__ == '__main__':
    Builder.load_string(KV_STRING)
    from test_maps import TEST_MAP_0, TEST_MAP_1, TEST_MAP_2
    TEST_MAPS = [TEST_MAP_0, TEST_MAP_1, TEST_MAP_2]

    CHECKPOINTS_FOLDER_DIR = os.path.join(os.getcwd(), "checkpoints")
    map_names = [el['name'] for el in TEST_MAPS]

    TARGET_MAP = 'two_hills_10x10'
    # TARGET_MAP = 'square_sink_10x10'
    # TARGET_MAP = 'wavelike_5x9'
    data_path = os.path.join(CHECKPOINTS_FOLDER_DIR, TARGET_MAP, TARGET_MAP + ".npy")
    SCKivyApp(data_path=data_path,
              to_animate=True,
              to_export=False,
              ).run()
