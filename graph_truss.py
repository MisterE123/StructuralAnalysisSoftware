import matplotlib.pyplot as plt

def plot_deformed_truss(ret, exaggeration=1.0):
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
    elem = ret['elem']
    displacements = ret['displacements_by_node']
    
    # Calculate deformed node coordinates
    deformed_nodes = nodes + (displacements * exaggeration)

    plt.figure(figsize=(10, 6))

    # Plot deformed truss shape (soft red, solid)
    for i, element in enumerate(elem):
        n1_idx = int(element[0])
        n2_idx = int(element[1])
        
        def_x = [deformed_nodes[n1_idx][0], deformed_nodes[n2_idx][0]]
        def_y = [deformed_nodes[n1_idx][1], deformed_nodes[n2_idx][1]]
        
        # 'indianred' or '#D9534F' are nice soft red colors
        label = 'Deformed' if i == 0 else None
        plt.plot(def_x, def_y, color='#D9534F', linewidth=2, zorder=1, label=label)

    # Plot original truss shape (grey, dashed)
    for i, element in enumerate(elem):
        n1_idx = int(element[0])
        n2_idx = int(element[1])
        
        orig_x = [nodes[n1_idx][0], nodes[n2_idx][0]]
        orig_y = [nodes[n1_idx][1], nodes[n2_idx][1]]
        
        label = 'Original' if i == 0 else None
        plt.plot(orig_x, orig_y, color='grey', linestyle='--', linewidth=2,zorder=2, label=label)


    # Marker scatter plot for the nodes (optional, helps visibility)
    plt.scatter(nodes[:, 0], nodes[:, 1], color='grey', zorder=3, s=20)
    plt.scatter(deformed_nodes[:, 0], deformed_nodes[:, 1], color='#D9534F', zorder=4, s=30)

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
                restr_x.append(nodes[i][0])
                restr_y.append(nodes[i][1])
                restr_u.append(-1)
                restr_v.append(0)
            if ry == 1:
                restr_x.append(nodes[i][0])
                restr_y.append(nodes[i][1])
                restr_u.append(0)
                restr_v.append(-1)
        if restr_x:
            # We use pivot='tip' so the arrow points TO the node
            plt.quiver(restr_x, restr_y, restr_u, restr_v, color='blue', pivot='tip', zorder=5, label='Supports')

    # Plot Applied Loads as green arrows
    app_loads = ret.get('app_loads')
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
            plt.quiver(load_x, load_y, load_u, load_v, color='green', zorder=6, label='Applied Loads')

    # Format the plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Truss Analysis: Original vs. Deformed Shape (Exaggeration={exaggeration}x)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.axis('equal') # Prevents distortion of the truss geometry
    
    plt.show()
