import panel as pn
import holoviews as hv
import param
import pandas as pd
import numpy as np

from stiffness_matrix import TrussModel2D

pn.extension(sizing_mode="stretch_width")
hv.extension('bokeh')

def point_line_distance(px, py, x1, y1, x2, y2):
    l2 = (x2 - x1)**2 + (y2 - y1)**2
    if l2 == 0: 
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    t = max(0, min(1, ((px - x1)*(x2 - x1) + (py - y1)*(y2 - y1)) / l2))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

class TrussApp(param.Parameterized):
    # Data DataFrames
    nodes_df = param.DataFrame(pd.DataFrame(columns=['label', 'x', 'y', 'restx', 'resty', 'loadx', 'loady', 'by_grid']))
    elements_df = param.DataFrame(pd.DataFrame(columns=['label', 'start', 'end', 'area', 'material']))
    materials_df = param.DataFrame(pd.DataFrame({'name': ['steel'], 'E': [29000.0]}))

    # Core State
    mode = param.Selector(default='Nodes and Members', objects=['Nodes and Members', 'Supports', 'Loads', 'Properties', 'Results'])
    active_tool = param.Selector(default='Select', objects=['Select', 'Draw Node', 'Draw Member', 'Delete'])
    
    grid_x = param.Number(10.0, bounds=(0.1, None))
    grid_y = param.Number(10.0, bounds=(0.1, None))
    snap_to_grid = param.Boolean(default=True)
    
    # Internal Canvas State
    current_start_node = param.String(None)
    status = param.String("Ready")
    analysis_results = param.Dict(default={})

    # Streams
    tap_stream = hv.streams.Tap(x=None, y=None)

    def __init__(self, **params):
        super().__init__(**params)
        self.tap_stream.param.watch(self._on_tap, ['x', 'y'])
        
        # Start with one default node
        self._add_node(0.0, 0.0)

    def _snap(self, val, step):
        if not self.snap_to_grid or step <= 0: return val
        return round(val / step) * step

    def _add_node(self, x, y):
        df = self.nodes_df.copy()
        new_idx = len(df)
        label = f"Node{new_idx+1}"
        df.loc[new_idx] = [label, float(x), float(y), 0.0, 0.0, 0.0, 0.0, True]
        self.nodes_df = df
        return label

    def _add_member(self, start_lbl, end_lbl):
        df = self.elements_df.copy()
        exists = df[((df['start']==start_lbl)&(df['end']==end_lbl)) | ((df['start']==end_lbl)&(df['end']==start_lbl))]
        if not exists.empty: return
        new_idx = len(df)
        label = f"elem{new_idx+1}"
        df.loc[new_idx] = [label, start_lbl, end_lbl, 8.0, "steel"]
        self.elements_df = df

    def find_nearest_node(self, x, y, threshold=None):
        if x is None or y is None or self.nodes_df.empty: return None
        if threshold is None: threshold = min(self.grid_x, self.grid_y) * 0.4
        best_dist, best_idx = float('inf'), None
        for idx, row in self.nodes_df.iterrows():
            d = np.sqrt((row['x'] - x)**2 + (row['y'] - y)**2)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        if best_dist < threshold: return best_idx
        return None

    def find_nearest_elem(self, x, y, threshold=None):
        if x is None or y is None or self.elements_df.empty or self.nodes_df.empty: return None
        if threshold is None: threshold = min(self.grid_x, self.grid_y) * 0.3
        node_lookup = {row['label']: (row['x'], row['y']) for _, row in self.nodes_df.iterrows()}
        best_dist, best_idx = float('inf'), None
        for idx, el in self.elements_df.iterrows():
            if el['start'] in node_lookup and el['end'] in node_lookup:
                s, e = node_lookup[el['start']], node_lookup[el['end']]
                d = point_line_distance(x, y, s[0], s[1], e[0], e[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
        if best_dist < threshold: return best_idx
        return None

    def _on_tap(self, *events):
        x, y = self.tap_stream.x, self.tap_stream.y
        if x is None or y is None: return
        
        nx, ny = x, y
        if self.snap_to_grid:
            nx = self._snap(x, self.grid_x)
            ny = self._snap(y, self.grid_y)
            
        if self.mode == 'Nodes and Members':
            if self.active_tool == 'Draw Node':
                n_idx = self.find_nearest_node(x, y)
                if n_idx is None:
                    self._add_node(nx, ny)
            
            elif self.active_tool == 'Draw Member':
                n_idx = self.find_nearest_node(x, y)
                if n_idx is not None:
                    lbl = self.nodes_df.at[n_idx, 'label']
                    if self.current_start_node is None:
                        self.current_start_node = lbl
                    else:
                        if self.current_start_node != lbl:
                            self._add_member(self.current_start_node, lbl)
                        self.current_start_node = None
                else:
                    self.current_start_node = None # cancel
                    
            elif self.active_tool == 'Delete':
                e_idx = self.find_nearest_elem(x, y)
                if e_idx is not None:
                    self.elements_df = self.elements_df.drop(e_idx).reset_index(drop=True)
                else:
                    n_idx = self.find_nearest_node(x, y)
                    if n_idx is not None:
                        df = self.nodes_df.drop(n_idx).reset_index(drop=True)
                        self.nodes_df = df
                        # Cascade delete
                        valid = set(df['label'].tolist())
                        edf = self.elements_df.copy()
                        if not edf.empty:
                            edf = edf[edf['start'].isin(valid) & edf['end'].isin(valid)].reset_index(drop=True)
                            self.elements_df = edf

    @param.depends('mode', watch=True)
    def _reset_tools(self):
        self.current_start_node = None
        if self.mode != 'Nodes and Members':
            self.active_tool = 'Select'
            
    # --- UI Generators ---
    
    @param.depends('nodes_df', 'elements_df', 'grid_x', 'grid_y', 'mode', 'active_tool', 'current_start_node')
    def plot_canvas(self):
        gx, gy = max(0.1, self.grid_x), max(0.1, self.grid_y)
        
        # Base Points
        pts_dict = {'x': [], 'y': [], 'label': []}
        if not self.nodes_df.empty: pts_dict = self.nodes_df.to_dict('list')
        pts = hv.Points(pts_dict, kdims=['x', 'y'], vdims=['label']).opts(
            size=14, color='gray', line_color='black', tools=['hover', 'tap'], active_tools=['tap']
        )
        self.tap_stream.source = pts
        
        # Base Lines
        lines = []
        node_lookup = {}
        if not self.nodes_df.empty:
            node_lookup = {row['label']: (row['x'], row['y']) for _, row in self.nodes_df.iterrows()}
        if not self.elements_df.empty:
            for _, el in self.elements_df.iterrows():
                if el['start'] in node_lookup and el['end'] in node_lookup:
                    lines.append([node_lookup[el['start']], node_lookup[el['end']]])
        paths = hv.Path(lines).opts(color='black', line_width=2)
        
        # Origin Axes
        ox = hv.Curve([(0, 0), (gx*2, 0)]).opts(color='gray', line_width=3)
        oy = hv.Curve([(0, 0), (0, gy*2)]).opts(color='gray', line_width=3)
        otx = hv.Text(gx*2 + gx*0.2, 0, "X").opts(color='black', text_font_size='12pt')
        oty = hv.Text(0, gy*2 + gy*0.2, "Y").opts(color='black', text_font_size='12pt')
        origin = ox * oy * otx * oty
        
        # Draw Preview State
        draw_pts = None
        if self.mode == 'Nodes and Members' and self.active_tool == 'Draw Member' and self.current_start_node in node_lookup:
            draw_pts = hv.Points([node_lookup[self.current_start_node]]).opts(size=18, color='orange', alpha=0.6)
            
        layout = (paths * pts * origin)
        if draw_pts: layout *= draw_pts
        
        max_x, max_y = gx*10, gy*10
        if not self.nodes_df.empty:
            max_x = max(max_x, self.nodes_df['x'].max() + gx*2)
            max_y = max(max_y, self.nodes_df['y'].max() + gy*2)
            
        xticks = [i * gx for i in range(-5, int(max_x / gx) + 5)]
        yticks = [i * gy for i in range(-5, int(max_y / gy) + 5)]
        
        return layout.opts(
            width=900, height=700, show_grid=True, xticks=xticks, yticks=yticks,
            xlim=(-gx*2, max_x), ylim=(-gy*2, max_y), data_aspect=1,
            toolbar=None, # Clean aesthetic
            bgcolor='#e8e8e8'
        )

    # --- Properties Sidebar Logic ---
    
    def _update_node_prop(self, idx, col, val):
        df = self.nodes_df.copy()
        df.at[idx, col] = val
        if col in ['x', 'y'] and self.snap_to_grid and df.at[idx, 'by_grid']:
            df.at[idx, col] = self._snap(float(val), self.grid_x if col=='x' else self.grid_y)
        self.nodes_df = df

    @param.depends('nodes_df', 'elements_df', 'mode')
    def get_properties_sidebar(self):
        if self.mode != 'Nodes and Members':
            return pn.Column(pn.pane.Markdown(f"*{self.mode} properties not active.*"))
            
        # Nodes
        node_items = [pn.pane.Markdown("### Nodes", margin=(0,0,10,0))]
        for idx, row in self.nodes_df.iterrows():
            w_name = pn.widgets.TextInput(value=row['label'], name="Name", width=120, margin=(0,5,0,0))
            w_name.param.watch(lambda e, i=idx: self._update_node_prop(i, 'label', e.new), 'value')
            
            w_bygrid = pn.widgets.Checkbox(value=row.get('by_grid', True), name="By grid", margin=(5,0,0,0))
            w_bygrid.param.watch(lambda e, i=idx: self._update_node_prop(i, 'by_grid', e.new), 'value')
            
            w_x = pn.widgets.FloatInput(value=float(row['x']), name="X", width=60, margin=(0,5,0,0))
            w_x.param.watch(lambda e, i=idx: self._update_node_prop(i, 'x', e.new), 'value')
            
            w_y = pn.widgets.FloatInput(value=float(row['y']), name="Y", width=60, margin=(0,0,0,0))
            w_y.param.watch(lambda e, i=idx: self._update_node_prop(i, 'y', e.new), 'value')
            
            card = pn.Column(
                pn.Row(w_name, w_bygrid),
                pn.Row(w_x, w_y),
                margin=(0,0,15,0)
            )
            node_items.append(card)
            
        node_items.append(pn.widgets.Button(name='[+]', width=50, button_type='light', on_click=lambda e: self._add_node(0,0)))
        
        # Members
        member_items = [pn.pane.Markdown("### Members", margin=(20,0,10,0))]
        node_lbls = self.nodes_df['label'].tolist() if not self.nodes_df.empty else []
        for idx, row in self.elements_df.iterrows():
            w_name = pn.widgets.TextInput(value=row['label'], name="Name", width=120)
            # Simplistic for prototype: just show text
            card = pn.Column(
                w_name,
                pn.pane.Markdown(f"start: [{row['start']}]<br>end: [{row['end']}]", style={'color': '#555'}),
                margin=(0,0,15,0)
            )
            member_items.append(card)
            
        return pn.Column(*(node_items + member_items), width=300, scroll=True, height=700, css_classes=['prop-sidebar'])


# --- CSS and Layout Assembly ---
css = """
.top-bar {
    background-color: #f2f2f2;
    border-bottom: 2px solid #5ab0f5;
    padding: 10px 20px;
}
.left-toolbar {
    background-color: #e2e2e2;
    padding: 15px;
}
.prop-sidebar {
    background-color: #555555;
    color: #e0e0e0;
    padding: 20px;
}
.prop-sidebar .bk-input {
    background-color: #777;
    color: white;
    border: none;
}
"""
pn.extension(raw_css=[css])

app = TrussApp()

# Top Bar Components
top_bar = pn.Row(
    pn.pane.Markdown("File", margin=(10,20,0,0)),
    pn.widgets.Select.from_param(app.param.mode, name="", width=250, margin=(5,20,0,0)),
    pn.layout.HSpacer(),
    pn.widgets.Checkbox.from_param(app.param.snap_to_grid, name="Snap", margin=(10,10,0,0)),
    pn.widgets.FloatInput.from_param(app.param.grid_x, name="Grid x:", width=70, margin=(5,5,0,0)),
    pn.widgets.FloatInput.from_param(app.param.grid_y, name="y:", width=70, margin=(5,20,0,0)),
    css_classes=['top-bar'], sizing_mode='stretch_width'
)

# Left Toolbar Components
# We use simple buttons to switch 'active_tool' param
def set_tool(event): app.active_tool = event.obj.name
b_sel = pn.widgets.Button(name='Select', width=50, height=50)
b_nod = pn.widgets.Button(name='Draw Node', width=50, height=50) # Prototype Icon representation
b_mem = pn.widgets.Button(name='Draw Member', width=50, height=50)
b_del = pn.widgets.Button(name='Delete', width=50, height=50)

b_sel.on_click(set_tool)
b_nod.on_click(set_tool)
b_mem.on_click(set_tool)
b_del.on_click(set_tool)

@param.depends(app.param.active_tool)
def get_toolbar(active):
    b_sel.button_type = 'primary' if active == 'Select' else 'light'
    b_nod.button_type = 'primary' if active == 'Draw Node' else 'light'
    b_mem.button_type = 'primary' if active == 'Draw Member' else 'light'
    b_del.button_type = 'primary' if active == 'Delete' else 'light'
    return pn.Column(b_sel, b_nod, b_mem, b_del, css_classes=['left-toolbar'], width=80, height=700)

main_area = pn.Row(
    get_toolbar,
    pn.panel(app.plot_canvas),
    pn.panel(app.get_properties_sidebar),
    sizing_mode='stretch_width'
)

dashboard = pn.Column(
    top_bar,
    main_area,
    sizing_mode='stretch_width',
    margin=0
)

dashboard.servable()
