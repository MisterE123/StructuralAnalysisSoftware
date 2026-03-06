import matplotlib.pyplot as plt

colors = {
    "tension":"#3978a8",
    "compression":"#f47e1b",
    "zero":"#cd6093",
    "original":"#5a5353",
    "supports":"#397b44",
    "loads":"#8e478c",
    "displaced_nodes":"#b6d53c"
}
def plot_deformed_truss(ret, exaggeration=1.0, force_visual_scale=1.0, zoom_factor=1.0):
    """
    Graphs the original and deformed shapes of a 2D truss structure.
    
    Parameters
    ----------
    ret : dict
        The result dictionary returned by TrussModel2D.run_analysis().
        Expected to contain:
            - "nodes": Original node coordinates, shape (N, 2)
            - "elem": Element connectivity list of node indices, shape (E, 2)
            - "displacements_by_node": Node displacements, shape (N, 2)
    exaggeration : float, optional
        A multiplier applied to the displacements to make them visually
        distinct. The default is 1.0.
    """
    nodes = ret['nodes']

    node_list = ret['nodes_list']
    elem_list = ret['elem_list']
    elem = ret['elem']
    normal_forces = ret['normal_forces']
    app_loads = ret['app_loads']
    forces_by_node = ret['forces_by_node']
    displacements = ret['displacements_by_node']
    
    # Calculate deformed node coordinates
    deformed_nodes = nodes + (displacements * exaggeration)

    # Compute graph height for scaling
    y_coords = nodes[:, 1]
    height = max(y_coords) - min(y_coords)
    default_arrow_len = height / 32

    # Compute average force magnitude for scaling quivers
    load_mags = []
    if app_loads is not None:
        for i in range(len(nodes)):
            lx = app_loads[i*2][0]
            ly = app_loads[i*2+1][0]
            if lx != 0 or ly != 0:
                mag = (lx**2 + ly**2)**0.5
                load_mags.append(mag)
    avg_mag = sum(load_mags) / len(load_mags) if load_mags else 1.0
    quiver_scale = (default_arrow_len / avg_mag) * force_visual_scale if avg_mag > 0 else force_visual_scale

    # apply custom grey palette for outlines and text
    grey_palette = ['#302c2e', '#5a5353', '#7d7071', '#a0938e', '#cfc6b8']
    plt.rcParams.update({
        'text.color': grey_palette[1],
        'axes.edgecolor': grey_palette[0],
        'axes.labelcolor': grey_palette[1],
        'xtick.color': grey_palette[2],
        'ytick.color': grey_palette[2],
        'grid.color': grey_palette[3],
        # figure background kept white for contrast
        'figure.facecolor': 'white'
    })
    plt.figure(figsize=(10, 6))

    # Plot deformed truss shape (soft red, solid)
    deformed_label_added = False
    for i, element in enumerate(elem):
        n1_idx = int(element[0])
        n2_idx = int(element[1])
        
        def_x = [deformed_nodes[n1_idx][0], deformed_nodes[n2_idx][0]]
        def_y = [deformed_nodes[n1_idx][1], deformed_nodes[n2_idx][1]]
        normal = normal_forces[i][0]
        is_tension = normal > 0
        is_zero = normal == 0

        # only label the first plotted element of each type once
        label = None
        if not deformed_label_added:
            label = 'Deformed'
            deformed_label_added = True

        if is_zero:
            plt.plot(def_x, def_y, color=colors['zero'], linewidth=2, zorder=1, label=label)
        elif is_tension:
            plt.plot(def_x, def_y, color=colors['tension'], linewidth=2, zorder=1, label=label)
        else:
            plt.plot(def_x, def_y, color=colors['compression'], linewidth=2, zorder=1, label=label)

    # Plot original truss shape (grey, dashed)
    original_label_added = False
    for i, element in enumerate(elem):
        n1_idx = int(element[0])
        n2_idx = int(element[1])
        
        orig_x = [nodes[n1_idx][0], nodes[n2_idx][0]]
        orig_y = [nodes[n1_idx][1], nodes[n2_idx][1]]
        
        label = None
        if not original_label_added:
            label = 'Original'
            original_label_added = True

        plt.plot(orig_x, orig_y, color=colors['original'], linestyle='--', linewidth=2,zorder=2, label=label)


    # Marker scatter plot for the nodes (optional, helps visibility)

    plt.scatter(nodes[:, 0], nodes[:, 1], color=colors['original'], zorder=3, s=20)
    plt.scatter(deformed_nodes[:, 0], deformed_nodes[:, 1], color=colors['displaced_nodes'], zorder=4, s=30)

    # Plot Support Conditions as blue arrows
    restr = ret.get('restr')
    if restr is not None:
        restr_x = []
        restr_y = []
        restr_u = []
        restr_v = []
        for i in range(len(nodes)):
            rx = restr[i*2][0]
            ry = restr[i*2+1][0]
            if rx == 1:
                fx = forces_by_node[i][0]
                restr_x.append(nodes[i][0])
                restr_y.append(nodes[i][1])
                restr_u.append(-abs(fx) * quiver_scale)
                restr_v.append(0)
            if ry == 1:
                fy = forces_by_node[i][1]
                restr_x.append(nodes[i][0])
                restr_y.append(nodes[i][1])
                restr_u.append(0)
                restr_v.append(-abs(fy) * quiver_scale)
        if restr_x:
            # We use pivot='tip' so the arrow points TO the node
            plt.quiver(restr_x, restr_y, restr_u, restr_v, color=colors['supports'], alpha=0.5, pivot='tip', zorder=5, label='Supports',)

    if app_loads is not None:
        load_x = []
        load_y = []
        load_u = []
        load_v = []
        for i in range(len(nodes)):
            lx = app_loads[i*2][0]
            ly = app_loads[i*2+1][0]
            if lx != 0 or ly != 0:
                load_x.append(nodes[i][0])
                load_y.append(nodes[i][1])
                load_u.append(lx)
                load_v.append(ly)
        if load_x:
            # Arrow points in the direction of the load, starting at the node
            plt.quiver(load_x, load_y, [u * quiver_scale for u in load_u], [v * quiver_scale for v in load_v], color=colors['loads'], alpha=0.5, pivot='tip', zorder=6, label='Applied Loads')

    # Format the plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Truss Analysis: Original vs. Deformed Shape (Exaggeration={exaggeration}x)')
    plt.grid(True, linestyle=':', alpha=0.7)

    # build a custom legend showing all color meanings
    from matplotlib.lines import Line2D
    legend_handles = []
    legend_labels = []

    # tension/compression/zero-force (deformed geometry)
    legend_handles.append(Line2D([0], [0], color=colors['tension'], lw=2))
    legend_labels.append('Tension (deformed)')
    legend_handles.append(Line2D([0], [0], color=colors['compression'], lw=2))
    legend_labels.append('Compression (deformed)')
    legend_handles.append(Line2D([0], [0], color=colors['zero'], lw=2))
    legend_labels.append('Zero-force member')

    # original shape
    legend_handles.append(Line2D([0], [0], color=colors['original'], linestyle='--', lw=2))
    legend_labels.append('Original shape')

    # displaced nodes
    legend_handles.append(Line2D([0], [0], marker='o', color=colors['displaced_nodes'], lw=0, markersize=8))
    legend_labels.append('Displaced node')

    # supports and loads
    legend_handles.append(Line2D([0], [0], color=colors['supports'], lw=2))
    legend_labels.append('Support reaction')
    legend_handles.append(Line2D([0], [0], color=colors['loads'], lw=2))
    legend_labels.append('Applied load')

    plt.legend(legend_handles, legend_labels, loc='best')
    plt.axis('equal') # Prevents distortion of the truss geometry

    # Apply zoom factor
    xlims = plt.xlim()
    ylims = plt.ylim()
    x_center = (xlims[0] + xlims[1]) / 2
    y_center = (ylims[0] + ylims[1]) / 2
    x_range = xlims[1] - xlims[0]
    y_range = ylims[1] - ylims[0]
    new_x_range = x_range / zoom_factor
    new_y_range = y_range / zoom_factor
    plt.xlim(x_center - new_x_range / 2, x_center + new_x_range / 2)
    plt.ylim(y_center - new_y_range / 2, y_center + new_y_range / 2)
    
    plt.show()

    # ---------- additional tables ----------
    # joint displacement table
    try:
        disp_data = displacements.tolist()
    except Exception:
        disp_data = [list(d) for d in displacements]
    disp_rows = []
    for idx, (ux, uy) in enumerate(disp_data):
        disp_rows.append([node_list[idx]["label"], f"{ux:.3g}", f"{uy:.3g}"])

    fig = plt.figure(figsize=(6, len(disp_rows)*0.3 + 1))
    fig.suptitle('Joint Displacements')
    tbl = plt.table(cellText=disp_rows,
                    colLabels=['Node', 'Ux', 'Uy'],
                    loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    plt.axis('off')
    plt.show()

    # support reaction table
    # highlight the support reactions' cells with the same color as the support arrows in the plot
    if restr is not None:
        react_rows = []
        for i in range(len(nodes)):
            rx = forces_by_node[i][0]
            ry = forces_by_node[i][1]
            react_rows.append([node_list[i]["label"], f"{rx:.3g}", f"{ry:.3g}"])
        fig = plt.figure(figsize=(6, len(react_rows)*0.3 + 1))
        fig.suptitle('Nodal Forces')
        tbl = plt.table(cellText=react_rows,
                        colLabels=['Node', 'Fx', 'Fy'],
                        loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        plt.axis('off')
        plt.show()

    # member axial force table
    try:
        nf_data = normal_forces.tolist()
    except Exception:
        nf_data = [list(nf) for nf in normal_forces]
    force_rows = []
    for idx, f in enumerate(nf_data):
        # assume scalar in first position
        val = f[0]
        force_rows.append([elem_list[idx]["label"], f"{val:.3g}"])
    fig = plt.figure(figsize=(6, len(force_rows)*0.3 + 1))
    fig.suptitle('Member Axial Forces (+ tension / - compression)')
    tbl = plt.table(cellText=force_rows,
                    colLabels=['Element', 'Axial Force'],
                    loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    plt.axis('off')
    plt.show()
