import panel as pn
import holoviews as hv
import param
import pandas as pd
import numpy as np

from stiffness_matrix import TrussModel2D

pn.extension('tabulator', sizing_mode="stretch_width")
hv.extension('bokeh')

def point_line_distance(px, py, x1, y1, x2, y2):
    """Calculate distance from point (px,py) to line segment (x1,y1)-(x2,y2)."""
    l2 = (x2 - x1)**2 + (y2 - y1)**2
    if l2 == 0: 
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    t = max(0, min(1, ((px - x1)*(x2 - x1) + (py - y1)*(y2 - y1)) / l2))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

class TrussApp(param.Parameterized):
    # --- Input DataFrames ---
    nodes_df = param.DataFrame(pd.DataFrame({
        'label': ['node1', 'node2', 'node3', 'node4', 'node5', 'node6'],
        'x': [0.0, 1.0, 2.0, 3.0, 1.0, 2.0],
        'y': [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        'restx': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'resty': [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        'loadx': [0.0, 0.0, 0.0, 0.0, 25.0, 0.0],
        'loady': [0.0, -75.0, 0.0, 0.0, 0.0, 60.0]
    }))
    
    materials_df = param.DataFrame(pd.DataFrame({
        'name': ['steel', 'aluminum'],
        'E': [29000.0, 10000.0]
    }))
    
    elements_df = param.DataFrame(pd.DataFrame({
        'label': ['elem1', 'elem2', 'elem3', 'elem4', 'elem5', 'elem6', 'elem7', 'elem8', 'elem9', 'elem10'],
        'start': ['node1', 'node2', 'node3', 'node5', 'node2', 'node3', 'node1', 'node2', 'node3', 'node4'],
        'end':   ['node2', 'node3', 'node4', 'node6', 'node5', 'node6', 'node5', 'node6', 'node5', 'node6'],
        'area':  [8.0, 8.0, 16.0, 8.0, 8.0, 8.0, 12.0, 12.0, 12.0, 16.0],
        'material': ['steel', 'steel', 'aluminum', 'steel', 'steel', 'steel', 'steel', 'steel', 'steel', 'aluminum']
    }))

    # --- App State ---
    grid_spacing_x = param.Number(288.0, bounds=(0.1, None), doc="Grid spacing X")
    grid_spacing_y = param.Number(216.0, bounds=(0.1, None), doc="Grid spacing Y")
    exaggeration = param.Number(100.0, bounds=(1, 1000), doc="Deformation scale multiplier")
    status = param.String("Ready")
    
    # --- UI Tool Modes ---
    tool_mode = param.Selector(default='Nodes', objects=['Nodes', 'Members', 'Supports', 'Loads', 'Properties'])
    paint_material = param.Selector(default='steel', objects=['steel', 'aluminum'])
    paint_area = param.Number(8.0, bounds=(0.01, None))
    apply_load_x = param.Number(0.0)
    apply_load_y = param.Number(0.0)
    
    # Internal state for drawing members
    current_start_node = param.String(None) 
    
    # --- Output DataFrames ---
    disp_result_df = param.DataFrame(pd.DataFrame({'Node Index': [0], 'dx': [0.0], 'dy': [0.0]}))
    force_result_df = param.DataFrame(pd.DataFrame({'Node Index': [0], 'Fx': [0.0], 'Fy': [0.0]}))
    normal_force_df = param.DataFrame(pd.DataFrame({'Element Index': [0], 'Normal Force': [0.0]}))
    
    # We will hold on to the last `ret` from run_analysis here to redraw efficiently
    last_analysis_ret = param.Dict(default={})

    # --- Actions ---
    solve_action = param.Action(lambda self: self._run_solve(), label='Solve Truss')
    
    # --- Tools & Interact ---
    def __init__(self, **params):
        super().__init__(**params)
        
        # Streams for interactive drawing
        self.node_stream = hv.streams.PointDraw(
            data=self.nodes_df[['x', 'y', 'label']].to_dict('list'),
            num_objects=100
        )
        self.node_stream.param.watch(self._sync_nodes, 'data')
        
        # Tap stream for interaction modes
        self.tap_stream = hv.streams.Tap(x=None, y=None)
        self.tap_stream.param.watch(self._on_tap, ['x', 'y'])
        
        self.param.watch(self._update_materials, 'materials_df')

    def _update_materials(self, event):
        mats = self.materials_df['name'].tolist()
        if mats:
            self.param.paint_material.objects = mats
            if self.paint_material not in mats:
                self.paint_material = mats[0]

    def _sync_nodes(self, event):
        """Update nodes DataFrame when nodes are drawn/moved in canvas."""
        if getattr(self, '_first_sync', True):
            self._first_sync = False
            return
            
        data = event.new
        if data and 'x' in data:
            current_df = self.nodes_df.copy()
            new_df = pd.DataFrame(data)
            
            # Merge while keeping existing columns like restx, loadx
            if len(new_df) > len(current_df):
                # Node added
                diff = len(new_df) - len(current_df)
                for i in range(diff):
                    new_idx = len(current_df) + i
                    new_df.loc[new_idx, 'label'] = f"node{new_idx+1}"
                    for col in ['restx', 'resty', 'loadx', 'loady']:
                        new_df.loc[new_idx, col] = 0.0
            
            # Update matching nodes natively
            for col in current_df.columns:
                if col not in new_df.columns and col in current_df:
                    # fill missing cols
                    new_df[col] = current_df[col]
            new_df = new_df.fillna(0)
            
            # Prevent updating if no changes occurred to avoid Tabulator render crashes
            if current_df.equals(new_df):
                return
                
            self.nodes_df = new_df

    def find_nearest_node(self, x, y, threshold=0.5):
        if x is None or y is None: return None
        best_dist = float('inf')
        best_idx = None
        for idx, row in self.nodes_df.iterrows():
            d = np.sqrt((row['x'] - x)**2 + (row['y'] - y)**2)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        if best_dist < threshold:
            return best_idx
        return None

    def find_nearest_elem(self, x, y, threshold=0.5):
        if x is None or y is None: return None
        node_lookup = {row['label']: (row['x'], row['y']) for _, row in self.nodes_df.iterrows()}
        best_dist = float('inf')
        best_idx = None
        for idx, el in self.elements_df.iterrows():
            if el['start'] in node_lookup and el['end'] in node_lookup:
                s = node_lookup[el['start']]
                e = node_lookup[el['end']]
                d = point_line_distance(x, y, s[0], s[1], e[0], e[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
        if best_dist < threshold:
            return best_idx
        return None

    def _on_tap(self, *events):
        # Depending on mode, handle canvas taps.
        x, y = self.tap_stream.x, self.tap_stream.y
        if x is None or y is None: 
            return
            
        threshold = 0.5 # Snapping threshold in logical grid coordinates
        
        if self.tool_mode == 'Members':
            n_idx = self.find_nearest_node(x, y, threshold)
            if n_idx is not None:
                lbl = self.nodes_df.at[n_idx, 'label']
                if self.current_start_node is None:
                    self.current_start_node = lbl
                else:
                    if self.current_start_node != lbl:
                        # Append new element
                        new_el = len(self.elements_df)
                        lbl_el = f"elem{new_el+1}"
                        # Adding element to dataframe
                        df = self.elements_df.copy()
                        df.loc[new_el] = [lbl_el, self.current_start_node, lbl, self.paint_area, self.paint_material]
                        self.elements_df = df
                    self.current_start_node = None
                    
        elif self.tool_mode == 'Supports':
            n_idx = self.find_nearest_node(x, y, threshold)
            if n_idx is not None:
                df = self.nodes_df.copy()
                rx = df.at[n_idx, 'restx']
                ry = df.at[n_idx, 'resty']
                # Cycle: Free(0,0)->Pinned(1,1)->RollerX(0,1)->RollerY(1,0)
                if rx == 0 and ry == 0:
                    rx, ry = 1.0, 1.0
                elif rx == 1 and ry == 1:
                    rx, ry = 0.0, 1.0
                elif rx == 0 and ry == 1:
                    rx, ry = 1.0, 0.0
                else:
                    rx, ry = 0.0, 0.0
                df.at[n_idx, 'restx'] = rx
                df.at[n_idx, 'resty'] = ry
                self.nodes_df = df
                
        elif self.tool_mode == 'Loads':
            n_idx = self.find_nearest_node(x, y, threshold)
            if n_idx is not None:
                df = self.nodes_df.copy()
                df.at[n_idx, 'loadx'] = self.apply_load_x
                df.at[n_idx, 'loady'] = self.apply_load_y
                self.nodes_df = df
                
        elif self.tool_mode == 'Properties':
            e_idx = self.find_nearest_elem(x, y, threshold)
            if e_idx is not None:
                df = self.elements_df.copy()
                df.at[e_idx, 'area'] = self.paint_area
                df.at[e_idx, 'material'] = self.paint_material
                self.elements_df = df

    @param.depends('tool_mode', watch=True)
    def _clear_start_node(self):
        self.current_start_node = None

    def _run_solve(self):
        self.status = "Solving..."
        try:
            m = TrussModel2D()
            m.set_xgrid(self.grid_spacing_x)
            m.set_ygrid(self.grid_spacing_y)
            
            # 1. Add definitions
            for _, row in self.materials_df.iterrows():
                m.add_material(str(row['name']), float(row['E']))
                
            for _, row in self.nodes_df.iterrows():
                m.add_node(
                    x=float(row['x']), y=float(row['y']), label=str(row['label']),
                    restx=int(row['restx']), resty=int(row['resty']),
                    loadx=float(row['loadx']), loady=float(row['loady']),
                    use_grid=True
                )
                
            for _, row in self.elements_df.iterrows():
                m.add_elem(
                    nodeStart=str(row['start']), nodeEnd=str(row['end']),
                    label=str(row['label']), area=float(row['area']),
                    materialname=str(row['material'])
                )
            
            # 2. Solve
            ret = m.run_analysis(debug=False)
            self.last_analysis_ret = ret
            
            # 3. Present data in output DFs
            nodes_len = len(ret['nodes'])
            elem_len = len(ret['elem'])
            
            disp = ret['displacements_by_node']
            self.disp_result_df = pd.DataFrame({
                'Node Index': range(nodes_len),
                'dx': disp[:, 0], 'dy': disp[:, 1]
            })
            
            forces = ret['forces_by_node']
            self.force_result_df = pd.DataFrame({
                'Node Index': range(nodes_len),
                'Fx': forces[:, 0], 'Fy': forces[:, 1]
            })
            
            n_forces = ret['normal_forces']
            self.normal_force_df = pd.DataFrame({
                'Element Index': range(elem_len),
                'Normal Force': n_forces[:, 0]
            })
            
            self.status = "Solved successfully."
            
        except Exception as e:
            self.status = f"Error: {str(e)}"

    @param.depends('nodes_df', 'elements_df', 'grid_spacing_x', 'grid_spacing_y', 'tool_mode', 'current_start_node')
    def plot_input_canvas(self):
        """HoloViews plot for interactive truss drawing."""
        gx = self.grid_spacing_x
        gy = self.grid_spacing_y
        
        # Tools logic
        tools = ['hover']
        active_tools = []
        if self.tool_mode == 'Nodes':
            active_tools = ['point_draw']
        else:
            tools.append('tap')
            active_tools = ['tap']

        # Plot points (nodes)
        pts = hv.Points(self.nodes_df, kdims=['x', 'y'], vdims=['label', 'restx', 'resty', 'loadx', 'loady'])
        pts = pts.opts(size=10, color='blue', tools=tools, active_tools=active_tools)
        
        # We must reassign streams safely
        if self.tool_mode == 'Nodes':
            self.node_stream.source = pts
        else:
            self.tap_stream.source = pts
        
        # Plot lines (elements)
        lines = []
        node_lookup = {row['label']: (row['x'], row['y']) for _, row in self.nodes_df.iterrows()}
        for _, el in self.elements_df.iterrows():
            if el['start'] in node_lookup and el['end'] in node_lookup:
                s = node_lookup[el['start']]
                e = node_lookup[el['end']]
                lines.append([(s[0], s[1]), (e[0], e[1])])
                
        element_paths = hv.Path(lines).opts(color='gray', line_width=2)
        
        # Active member drawing visualization
        draw_pts = None
        if self.tool_mode == 'Members' and self.current_start_node and self.current_start_node in node_lookup:
            node_coord = node_lookup[self.current_start_node]
            draw_pts = hv.Points([node_coord]).opts(size=14, color='orange')
        
        # Show support conditions
        supports = []
        for _, n in self.nodes_df.iterrows():
            if n['restx'] or n['resty']:
                supports.append((n['x'], n['y']))
                
        supp_pts = hv.Points(supports).opts(size=14, marker='square', color='red', fill_alpha=0)
        
        # Show loads
        load_vectors = []
        for _, n in self.nodes_df.iterrows():
            lx, ly = n['loadx'], n['loady']
            if lx != 0 or ly != 0:
                # Quiver uses x, y, angle, magnitude
                angle = np.arctan2(ly, lx)
                mag = np.sqrt(lx**2 + ly**2)
                # Just draw a dot for loads on input since HoloViews quiver is sometimes tricky
                load_vectors.append((n['x'], n['y']))
        load_pts = hv.Points(load_vectors).opts(size=10, marker='triangle', color='green')

        layout = (element_paths * pts * supp_pts * load_pts)
        if draw_pts:
            layout = layout * draw_pts
            
        return layout.opts(
            width=600, height=400, title=f"Input Canvas (Mode: {self.tool_mode})",
            xaxis=None, yaxis=None, show_grid=True
        )
        
    @param.depends('last_analysis_ret', 'exaggeration')
    def plot_deformed_shape(self):
        """HoloViews plot for the final deformed shape."""
        ret = self.last_analysis_ret
        if not ret or 'nodes' not in ret:
            return hv.Text(0.5, 0.5, "Solve model to view results").opts(width=600, height=400)
            
        nodes = ret['nodes']
        elem = ret['elem']
        disp = ret['displacements_by_node']
        ex = self.exaggeration
        
        def_nodes = nodes + (disp * ex)
        
        # Original paths
        orig_lines = []
        for el in elem:
            s, e = int(el[0]), int(el[1])
            orig_lines.append([nodes[s], nodes[e]])
        orig_paths = hv.Path(orig_lines).opts(color='gray', line_dash='dashed', line_width=2)
        
        # Deformed paths
        def_lines = []
        for el in elem:
            s, e = int(el[0]), int(el[1])
            def_lines.append([def_nodes[s], def_nodes[e]])
        def_paths = hv.Path(def_lines).opts(color='#D9534F', line_width=3)
        
        orig_pts = hv.Points(nodes).opts(color='gray', size=5)
        def_pts = hv.Points(def_nodes).opts(color='#D9534F', size=8)
        
        layout = (orig_paths * orig_pts * def_paths * def_pts).opts(
            width=600, height=400, title=f"Deformed Shape (Scale = {ex}x)",
            show_grid=True, data_aspect=1
        )
        return layout

# --- Layout ---
app = TrussApp()

# Widgets
w_grid_x = pn.widgets.FloatInput.from_param(app.param.grid_spacing_x)
w_grid_y = pn.widgets.FloatInput.from_param(app.param.grid_spacing_y)
w_solve = pn.widgets.Button.from_param(app.param.solve_action, button_type='primary')
w_status = pn.widgets.StaticText.from_param(app.param.status)
w_exag = pn.widgets.FloatSlider.from_param(app.param.exaggeration)

w_tool = pn.widgets.RadioButtonGroup.from_param(app.param.tool_mode, button_type='success', sizing_mode='stretch_width')

@pn.depends(app.param.tool_mode)
def get_tool_sidebar(mode):
    if mode == 'Properties':
        return pn.Column(
            pn.pane.Markdown("**Paint Area/Material on existing members:**"),
            pn.widgets.Select.from_param(app.param.paint_material),
            pn.widgets.FloatInput.from_param(app.param.paint_area)
        )
    elif mode == 'Loads':
        return pn.Column(
            pn.pane.Markdown("**Tap node to apply this load:**"),
            pn.widgets.FloatInput.from_param(app.param.apply_load_x, name="Load X"),
            pn.widgets.FloatInput.from_param(app.param.apply_load_y, name="Load Y")
        )
    elif mode == 'Supports':
        return pn.pane.Markdown("**Tap node to cycle supports (Free -> Pinned -> Roller-X -> Roller-Y).**")
    elif mode == 'Members':
        return pn.pane.Markdown("**Tap two nodes sequentially to draw an element connecting them.**")
    elif mode == 'Nodes':
        return pn.pane.Markdown("**Click empty space to create nodes or drag existing ones.**")
    return pn.pane.Markdown("")

# Tables
t_materials = pn.widgets.Tabulator(app.param.materials_df, height=150)
t_nodes = pn.widgets.Tabulator(app.param.nodes_df, height=250)
t_elements = pn.widgets.Tabulator(app.param.elements_df, height=250)

t_res_disp = pn.widgets.Tabulator(app.param.disp_result_df, disabled=True, height=200)
t_res_force = pn.widgets.Tabulator(app.param.force_result_df, disabled=True, height=200)
t_res_nf = pn.widgets.Tabulator(app.param.normal_force_df, disabled=True, height=200)

input_tab = pn.Row(
    pn.Column(
        pn.pane.Markdown("### Structure Definition"),
        pn.Row(w_grid_x, w_grid_y),
        pn.pane.Markdown("**Materials**"), t_materials,
        pn.pane.Markdown("**Nodes**"), t_nodes,
        pn.pane.Markdown("**Elements**"), t_elements,
        width=500
    ),
    pn.Column(
        pn.pane.Markdown("### Interactive Canvas"),
        w_tool,
        pn.panel(get_tool_sidebar),
        pn.panel(app.plot_input_canvas)
    )
)

results_tab = pn.Row(
    pn.Column(
        pn.pane.Markdown("### Results Dashboard"),
        w_exag,
        pn.panel(app.plot_deformed_shape)
    ),
    pn.Column(
        pn.pane.Markdown("**Displacements**"), t_res_disp,
        pn.pane.Markdown("**Reactions/Forces**"), t_res_force,
        pn.pane.Markdown("**Axial Forces**"), t_res_nf,
        width=400
    )
)

main_layout = pn.Column(
    pn.pane.Markdown("# Interactive 2D Truss Solver"),
    pn.Row(w_solve, w_status),
    pn.Tabs(
        ("Input", input_tab),
        ("Results", results_tab)
    )
)

main_layout.servable()
