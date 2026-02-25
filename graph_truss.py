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

    # Format the plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Truss Analysis: Original vs. Deformed Shape (Exaggeration={exaggeration}x)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.axis('equal') # Prevents distortion of the truss geometry
    
    plt.show()
