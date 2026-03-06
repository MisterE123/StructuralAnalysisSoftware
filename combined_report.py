# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 17:50:12 2026

Truss analysis library.

@author: Gaudeor Rudmin
@licesne: MIT License

This program uses code and is based on code written by
Dr. Steven Woodruff
at James Madison University,
Released under the MIT License.


Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to deal 
in the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.


AI use: AI was used for minimal bug identification and organization suggestions
Mostly as a glorified api lookup.

"""

import numpy as np
import math
import matplotlib.pyplot as plt



def num_nodes_from_elem(elem):
    return int(np.max(elem)) + 1

def check_stable(S_ff):
    """
    Checks the stability of the truss by attempting to perform a Cholesky decomposition
    on the free-free stiffness matrix S_ff. If S_ff is not positive definite, it
    raises a ValueError indicating that the truss is unstable.
    """
    try:
        np.linalg.cholesky(S_ff)
    except np.linalg.LinAlgError:
        raise ValueError("Truss Is Unstable")

def element_global_stiffness(nodes, elem, elast, areas):

    # a) determine the num elemes in the structure

    num_elem = elem.shape[0]
    # print("There are "+str(num_elem)+" bars")
    
    K_list = [] # list of global stiffness matrs
    
    T_list = []

    for m in range(num_elem):
        
        start_node = elem[m,0]
        end_node = elem[m,1]
        X_start = nodes[start_node, 0]
        Y_start = nodes[start_node, 1]
        X_end = nodes[end_node, 0]
        Y_end = nodes[end_node, 1]
        dX = X_end - X_start
        dY = Y_end - Y_start
        L = np.sqrt(dX**2 + dY**2)
        E = elast[m]
        A = areas[m]
        
        k_local_base = np.array([[1,0,-1,0],
                                [0,0,0,0],
                                [-1,0,1,0],
                                [0,0,0,0]])

        # local stiffness matrix
        k_local = (E*A/L)*k_local_base
        
        c = dX/L # cos (theta)
        s = dY/L # sin (theta)
        
        T = np.array([[c,s,0,0],
                      [-s,c,0,0],
                      [0,0,c,s],
                      [0,0,-s,c]])
        TT = T.T
        
        K_m = TT @ k_local @ T #local transformed to global
        
        K_list.append(K_m)
        T_list.append(T)
        
        # we now have a list of member stiffness matricies K in 
        # global coords. Next, we must assemble the final global
        # stiffness matrix (see assemble_and_partition)

    return K_list, T_list


def is_restricted(dof_idx, restrained_dof):
    return (restrained_dof[0][dof_idx] == 1)

def count_unrestrained_dof(restrained_dofs):
    # count the number of unrestrained dof
    num_unrestr = 0
    
    
    for restr_row in restrained_dofs:
        if restr_row[0] == 0:
            num_unrestr += 1
    
    return num_unrestr

def get_free_and_restr_idxs(restrained_dofs):
    free_dof_idxs = []
    restr_dof_idxs = []
    for dof_idx in range(len(restrained_dofs)):
        if restrained_dofs[dof_idx][0] == 1:
            restr_dof_idxs.append(dof_idx)
        else:
            free_dof_idxs.append(dof_idx)
    return free_dof_idxs, restr_dof_idxs


def get_subvector(v,idxs_to_keep):
    # v[choose these rows, choose these cols (:=all)]
    return v[np.asarray(idxs_to_keep),:]
    
    
def assemble_and_partition(K_members, restrained_dofs, elem):
    ##
    ## Part A: assemble the full structural stiffness matrix
    ##
    
    # this gets the largest node idx from the element list
    # a clever way to extract the number of nodes without
    # actually passing the node list
    num_nodes = num_nodes_from_elem(elem)
    
    ndof = 2* num_nodes
    
    # num_unrestr = count_unrestrained_dof(restrained_dofs)

    # print("Sanity check: ", ndof == len(restrained_dofs))

    # make ndof x ndof matrix for the main stiffness matrix
    
    S = np.zeros((ndof,ndof))
    
    # extract the number of elements by the length of the 
    # element list
    
    for m in range(len(elem)):
        
        i = elem[m,0] # first node idx of the element (start)
        j = elem[m,1] # end node idx of the element
        
        K = K_members[m] # collect the member's Stiffness matrix
        
        # determine the dof list idxs for each member
        dof_ix = 2*i 
        # corresponds to the 0th idx in K
        dof_iy = 2*i + 1
        # corresponds to the 1st idx in K
        dof_jx = 2*j
        # corresponds to the 2nd idx in K
        dof_jy = 2*j + 1
        # corresponds to the 3rd idx in K
        
        member_dofs = [dof_ix, dof_iy, dof_jx, dof_jy]
        
        # our approach will be to add each member matrix into the main
        # stiffness matrix
        
        # h and k take on the idx of K, and we collect the idxs of
        # S from the member_dofs list
        for h in range(4):
            for k in range(4):
                Sy = member_dofs[h]
                Sx = member_dofs[k]
                
                S[Sy][Sx] += K[h][k]
                
    # now we have a Full S stiffness matrix
    # print(S)
    
    # next we need to remove unrestrained DOFs to get S_ff
    # first copy S into S_ff and S_rf
    

    free_dof_idxs, restr_dof_idxs = get_free_and_restr_idxs(restrained_dofs)
    #delete restrained rows leaving free
    temp_arr = np.delete(S, restr_dof_idxs, axis=0) 
    #delete restrained cols leaving free
    S_ff = np.delete(temp_arr, restr_dof_idxs, axis=1) 
    #check that the truss is in fact stable
    check_stable(S_ff)
    #delete free rows leaving restr
    temp_arr_2 = np.delete(S, free_dof_idxs, axis=0) 
    #delete restrained cols leaving free
    S_rf = np.delete(temp_arr_2, restr_dof_idxs, axis=1) 
    # we now have S, S_ff, and S_rf
    return S, S_ff, S_rf


def get_free_loads(P,restrained_dofs):
    free_idxs, restr_idxs = get_free_and_restr_idxs(restrained_dofs)
    return get_subvector(P, free_idxs)


def get_displacements(S_ff,P_f): 
    S_ff_inv = np.linalg.inv(S_ff)
    return S_ff_inv @ P_f

def get_support_reactions(S_rf,d):
    return S_rf @ d

# d_f: the free-dof displacement vector. Gives a shape (ndof,1) numpy array 
# that is the displacements.
def assemble_full_displacement_vector(d_f,restrained_dofs):
    """
    

    Parameters
    ----------
    d_f : the free-dof displacement vector 
            Gives a shape (ndof,1) numpy array.
    restrained_dofs : VECTOR OF RESTRAINED DOF.

    Returns
    -------
    v : DOF-BASED (ndof,1) NUMPY ARRAY OF DISPLACEMENTS.

    """
    free_idxs, restr_idxs = get_free_and_restr_idxs(restrained_dofs)
    v = np.zeros((len(restrained_dofs),1))
    free_dof_cnt = 0
    for free in free_idxs:
        v[free][0] = d_f[free_dof_cnt][0]
        free_dof_cnt += 1 
    return v



def assemble_full_load_vector(P_f, P_r, restrained_dofs):
    """
    

    Parameters
    ----------
    P_f : Free-dof loading vector.
    P_r : restrained dof loading vector.
    restrained_dofs : TYPE
        DESCRIPTION.

    Returns
    -------
    v : a DOF-BASED (ndof,1) numpy array of loads.

    """
    free_idxs, restr_idxs = get_free_and_restr_idxs(restrained_dofs)
    v = np.zeros((len(restrained_dofs),1))
    dof_cnt = 0
    free_dof_cnt = 0 
    restr_dof_cnt = 0
    for dof_row in restrained_dofs:
        is_free = (dof_row[0] == 0)
        if is_free:
            v[dof_cnt][0] = P_f[free_dof_cnt][0]
            free_dof_cnt += 1
        else:
            v[dof_cnt][0] = P_r[restr_dof_cnt][0]
            restr_dof_cnt += 1
        dof_cnt += 1
    return v

def convert_dof_v_to_np_node_v(dof_v):
    """
    
    Parameters
    ----------
    dof_v : AN Nx1 DOF-BASED NP ARRAY VECTOR.

    Returns
    -------
    node_val_list : A (N/2)x2 NP ARRAY OF NODE-BASED VALUES.

    """
    num_nodes = int(len(dof_v)/2)
    # print(num_nodes)
    node_val_list = np.zeros((num_nodes,2))
    dof_cnt = 0
    for node_idx in range(num_nodes):
        node_val_list[node_idx][0] = dof_v[dof_cnt][0]
        dof_cnt += 1
        node_val_list[node_idx][1] = dof_v[dof_cnt][0]
        dof_cnt += 1
    return node_val_list


def get_global_displacement_vectors_by_elem(elem,full_displacement_vector):
    """
    returns A per-element list of global-cs displacement vectors 
    [[[s_x],[s_y],[e_x],[e_y]],...]

    Parameters
    ----------
    elem : element list.
    full_displacement_vector : displacements by DOF

    Returns
    -------
    v_vector : list of global end deformations by element.
    [[[s_x], # elem 1 displacement vector
      [s_y],
      [e_x],
      [e_y]],
     ... # more elem displacement vectors
     ]
    
    """
    v_vector = np.zeros((elem.shape[0],4,1))
    
    np_displacements = convert_dof_v_to_np_node_v(full_displacement_vector)
    
    
    for elem_cnt in range(elem.shape[0]):
        element = elem[elem_cnt]
        elem_n1 = element[0]
        elem_n2 = element[1]
        
        n1_dx = np_displacements[elem_n1][0]
        n1_dy = np_displacements[elem_n1][1]
        n2_dx = np_displacements[elem_n2][0]
        n2_dy = np_displacements[elem_n2][1]
        
        v_vector[elem_cnt][0][0] = n1_dx
        v_vector[elem_cnt][1][0] = n1_dy
        v_vector[elem_cnt][2][0] = n2_dx
        v_vector[elem_cnt][3][0] = n2_dy
        
    # print(v_vector)

    return v_vector

def get_local_deformations_by_elem(v_vector, T_list):
    """
    
    Parameters
    ----------
    v_vector : Nx4x1 shape np array of N-elem based deformations.
    [[[s_x], # elem 1 displacement vector
      [s_y],
      [e_x],
      [e_y]],
     ... # more elem displacement vectors
     ]
    T_list : list of local transformation matricies per element.
    [[ 4 x 4 np matrix ], elem 1 transform matrix
     [ 4 x 4 np matrix ], # more transform matricies
    ]

    Raises
    ------
    IndexError
        If the number of transformations and element displacements passed are
        unequal.

    Returns
    -------
    u_vector : an Nx4x1 shape np vector of per-elem local displacement vectors
    [[[s_x], # elem 1 displacement vector
      [s_y],
      [e_x],
      [e_y]],
     ... # more elem displacement vectors
     ].

    """
    
    # sanity check
    if not v_vector.shape[0] == len(T_list):
        raise IndexError()
    
    u_vector = np.zeros((v_vector.shape[0],4,1))
    
    for idx in range(v_vector.shape[0]):
        
        u_vector[idx] = ((T_list[idx])@(v_vector[idx]))
    
    return u_vector


def get_normal_force(u,E,A,L):
    """
    
    Parameters
    ----------
    u : 4x1 elem-based local deformation vector.
    
    E : Youngs mod.
    
    A : cross-sectional area.
    
    L : Member original len.

    Returns
    -------
    N : Normal Force.

    """
    
    N = (u[2][0]-u[0][0])*E*A/L
    
    return N

def get_normal_force_list(u_vector, elast, areas, nodes, elem):
    
    L_list = []
    u_vect_len = u_vector.shape[0]
    
    for idx in range(u_vect_len):
        # get the original length
        element = elem[idx]
        elem_n1 = element[0]
        elem_n2 = element[1]
        
        n1x = nodes[elem_n1][0]
        n1y = nodes[elem_n1][1]
        
        n2x = nodes[elem_n2][0]
        n2y = nodes[elem_n2][1]
        
        L = math.sqrt((n2x-n1x)*(n2x-n1x)+(n2y-n1y)*(n2y-n1y))
        L_list.append(L)
    
    # calculate the Normal forces
    N_vector = np.zeros((u_vect_len,1))
    for idx in range(u_vect_len):
        u = np.asarray(u_vector[idx])
        E = elast[idx][0]
        A = areas[idx][0]
        L = L_list[idx]
        
        N_vector[idx][0] = get_normal_force(u, E, A, L)
    return N_vector

def analyze_model(nodes, elem, elast, areas, restr, app_loads, debug=False):
    if debug:
        print("nodes= ")
        print(nodes)
        print("")
        print("elem= ")
        print(elem)
        print("")
        print("elast= ")
        print(elast)
        print("")
        print("areas= ")
        print(areas)
        print("")
        print("restr= ")
        print(restr)
        print("")
        print("app_loads= ")
        print(app_loads)
        
    K_list, T_list = element_global_stiffness(nodes, elem, elast, areas)
    
    if debug:
        for i, T in enumerate(T_list):
            print("T(",i,") = ")
            print(T)
        for i, K in enumerate(K_list):
            print("K(",i,") = ")
            print(K)
        
    S, S_ff, S_rf = assemble_and_partition(K_list, restr, elem)

    if debug:
        print("")
        print("S= ")
        print(S)
        print("")
        print("S_ff= ")
        print(S_ff)
        print("")
        print("S_rf= ")
        print(S_rf)
        
    P_f = get_free_loads(app_loads, restr)

    if debug:
        print("")
        print("P_f= ")
        print(P_f)
        
    d_f = get_displacements(S_ff,P_f)
    
    
    if debug:
        print("")
        print("d_f= ")
        print(d_f)
        
    P_r = get_support_reactions(S_rf, d_f)

    if debug:
        print("")
        print("P_r= ")
        print(P_r)
    
    displacements = assemble_full_displacement_vector(d_f, restr)
    forces = assemble_full_load_vector(P_f, P_r, restr)
    
    displacements_by_node = convert_dof_v_to_np_node_v(displacements)
    forces_by_node = convert_dof_v_to_np_node_v(forces)

    if debug:
        print("displacements = ")
        print(displacements)
    
        print("forces = ")
        print(forces)
    
    # a list of node displacement lists [[dx,dy]...]
    per_elem_disp_list = get_global_displacement_vectors_by_elem(elem,
                                                                 displacements)
    
    if debug:
        print("per-elem global displacements = ")
        print(per_elem_disp_list)
    
    per_elem_local_disp_list = get_local_deformations_by_elem(per_elem_disp_list,
                                                              T_list)
    
    if debug:
        print("per-elem local displacements = ")
        print(per_elem_local_disp_list)
        
    per_elem_normal_force = get_normal_force_list(per_elem_local_disp_list, 
                                                  elast, 
                                                  areas, 
                                                  nodes, 
                                                  elem)
    if debug:
        print("per-elem Normal forces = ")
        print(per_elem_normal_force)
    
    return displacements_by_node, forces_by_node, per_elem_normal_force




"""
The following code contains the AI generated and human reviewed visualization code
"""


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

"""
The following code contains the easy-to use API for defining and analyzing a truss model
"""

class TrussModel2D:
    """
    Human-readable model state class.
    Stores Human-readable model data, provides methods
    to build the model, and provides a method to solve.
    
    usage:
        m = TrussModel2D()
        m.set_xgrid(xgrid_spacing)
        m.set_ygrid(ygrid_spacing)
        m.add_node(x=0,y=0,             # the coordinates of the node in GCS 
                                        # or grid coords
                                        # depending on use_grid
                   label="node1",       # Human-readable node name
                   restx=1,resty=1,     # restraints: 1 = restrained, 0 = free
                   loadx=25,loady=30,   # applied loading
                   use_grid=True        # whether to use grid for x,y
                   )
        # (add more nodes)
        
        m.add_material("steel", # human-readable material name
                       29000    # E, youngs modulus
                       )

        # (add more materials)
        
        m.add_elem(nodeStart      = "node1", # label of element start node
                  nodeEnd        = "node2",  # label of element end node
                  label          = "elem1",  # label of element
                  area           = 8, #in^2  # cross-sectional area of element
                  materialname   = "steel"   # name of element's material
                  )
        ret = m.run_analysis(debug=True) # debug = True will print all intermediate data. 
        # ret will have 
        #        {"nodes":nodes,
        #        "restr":restr,
        #        "app_loads":app_loads,
        #        "elem":elem,
        #        "elast":elast,
        #        "areas":areas,
        #        "displacements_by_node":displacements_by_node,
        #        "forces_by_node":forces_by_node,
        #        "normal_forces":normal_forces
        #        }
    """
    def __init__(self, xgrid=1, ygrid=1):
        self.xgrid = float(xgrid)
        self.ygrid = float(ygrid)
        self.nodeList = []
        self.elemList = []
        self.matrList = []
        self.add_material("default", 29000) # a default steel material
    
    def set_xgrid(self, xgrid):
        self.xgrid = xgrid
    def set_ygrid(self, ygrid):
        self.ygrid = ygrid
    def add_node(self, x, y, label, restx=0, resty=0, loadx=0, loady=0, 
                 use_grid=True):
        if use_grid:
            x *= self.xgrid
            y *= self.ygrid
        idx0 = len(self.nodeList)
        self.nodeList.append({
            "x":x,
            "y":y,
            "idx0":idx0,
            "label":label,
            "restx":restx,
            "resty":resty,
            "loadx":loadx,
            "loady":loady
            })
    def add_material(self,name,youngMod):
        self.matrList.append({
            "name": name, # Material string, idx 1
            "youngMod": youngMod # E in [ksi]
            })
    def findNodeIdxByLabel(self, label):
        for nodeData in self.nodeList:
            if nodeData["label"] == label:
                return nodeData["idx0"]
    def add_elem(self,nodeStart,nodeEnd,label,area=1,materialname="default"):
        snodeidx = 0
        if isinstance(nodeStart, str):
            snodeidx = self.findNodeIdxByLabel(nodeStart)
        else:
            snodeidx = nodeStart
            
        enodeidx = 0
        if isinstance(nodeEnd, str):
            enodeidx = self.findNodeIdxByLabel(nodeEnd)
        else:
            enodeidx = nodeEnd
                
        idx0 = len(self.elemList)
        
        self.elemList.append({
            "start":snodeidx,
            "end":enodeidx,
            "label":label,
            "idx0":idx0,
            "area":area,
            "material_name":materialname
        })
    def parseNodesIntoDOFList(self):
        restr = np.zeros((len(self.nodeList)*2,1),dtype=int)
        app_loads = np.zeros((len(self.nodeList)*2,1),dtype=float)
        
        idx = 0
        for nodeDict in self.nodeList:
            
            restr[idx*2] = nodeDict["restx"]
            restr[idx*2+1] = nodeDict["resty"]
            app_loads[idx*2] = nodeDict["loadx"]
            app_loads[idx*2+1] = nodeDict["loady"]
            
            idx += 1
            
        return restr, app_loads

    def getMaterialByName(self,name):
        for materialData in self.matrList:
            if materialData["name"] == name:
                return materialData

    def getMaterialEByName(self,name):
        matrl = self.getMaterialByName(name)
        return matrl["youngMod"]

    def run_analysis(self,debug=False):
        
        nodes = np.array([[self.nodeList[idx]["x"], self.nodeList[idx]["y"]] 
                          for idx in range(len(self.nodeList))],dtype=float)
        
        restr, app_loads = self.parseNodesIntoDOFList()
        
        elem = np.array([[elemData["start"],elemData["end"]] for elemData in 
                         self.elemList],dtype=int)
        elast = np.array([[self.getMaterialEByName(elemData["material_name"])] 
                          for elemData in self.elemList])
        areas = np.array([[elemData["area"]] for elemData in self.elemList])

        displacements_by_node, forces_by_node, normal_forces = analyze_model(
                                              nodes, elem, 
                                              elast, areas, 
                                              restr, app_loads, 
                                              debug)
        ret = {"nodes":nodes,
               "restr":restr,
               "app_loads":app_loads,
               "elem":elem,
               "elast":elast,
               "areas":areas,
               "displacements_by_node":displacements_by_node,
               "forces_by_node":forces_by_node,
               "normal_forces":normal_forces,
               "nodes_list":self.nodeList,
               "elem_list":self.elemList,
               "matr_list":self.matrList
               }
        
        return ret
   

     
def main():
    
    """
        Running the file by itself will run the following code
        
        This solves the example truss
        
        Passing debug = True like we do will print all intermediate data
        
    """
    T1 = TrussModel2D()
    T1.set_xgrid(8*1000) # mm
    T1.set_ygrid(6*1000) # mm
    T1.add_node(x=0,y=0,             # the coordinates of the node in GCS 
                                    # or grid coords
                                    # depending on use_grid
            label="node1",          # Human-readable node name
            restx=1,resty=1,        # restraints: 1 = restrained, 0 = free
            loadx=0,loady=0,        # applied loading
            use_grid=True           # whether to use grid for x,y
            )
    # (add more nodes)
    T1.add_node(x=1,y=0, label="node2", restx=1,resty=1, loadx=0,loady=0,    use_grid=True)

    T1.add_node(x=0,y=1, label="node3", restx=0,resty=0, loadx=0,loady=0,    use_grid=True)
    T1.add_node(x=1,y=1, label="node4", restx=0,resty=0, loadx=6,loady=0,    use_grid=True)
    T1.add_node(x=0,y=2, label="node5", restx=0,resty=0, loadx=0,loady=-20,  use_grid=True)
    T1.add_node(x=1,y=2, label="node6", restx=0,resty=0, loadx=12,loady=-20, use_grid=True)

    T1.add_material("matrl1", # human-readable material name
                    200    # E, youngs modulus, Gpa
                    )

    # (add more materials)

    A = 1000 # mm^2, cross-sectional area of elements
    T1.add_elem(nodeStart       = "node1", # label of element start node
                nodeEnd        = "node3",  # label of element end node
                label          = "elem1",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    T1.add_elem(nodeStart       = "node2", # label of element start node
                nodeEnd        = "node4",  # label of element end node
                label          = "elem2",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    T1.add_elem(nodeStart       = "node3", 
                nodeEnd        = "node5",  # label of element end node
                label          = "elem3",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    T1.add_elem(nodeStart       = "node4",
                nodeEnd        = "node6",  # label of element end node
                label          = "elem4",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    T1.add_elem(nodeStart       = "node3",
                nodeEnd        = "node4",  # label of element end node
                label          = "elem5",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    T1.add_elem(nodeStart       = "node5",
                nodeEnd        = "node6",  # label of element end node
                label          = "elem6",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    T1.add_elem(nodeStart       = "node1",
                nodeEnd        = "node4",  # label of element end node
                label          = "elem7",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    T1.add_elem(nodeStart       = "node2",
                nodeEnd        = "node3",  # label of element end node
                label          = "elem8",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    T1.add_elem(nodeStart       = "node3",
                nodeEnd        = "node6",  # label of element end node
                label          = "elem9",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    T1.add_elem(nodeStart       = "node4",
                nodeEnd        = "node5",  # label of element end node
                label          = "elem10",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )


     # Run analysis and get results


    ret1 = T1.run_analysis(debug=True) # debug = True will print all intermediate data. 
    plot_deformed_truss(ret1, exaggeration=100, force_visual_scale = .5, zoom_factor = .8) # exaggeration is a multiplier to make the displacements visually distinct

    """
    Now we will solve the second test truss
    """

    T2 = TrussModel2D()
    T2.set_xgrid(3*12) # in
    T2.set_ygrid(6*12) # in
    T2.add_node(x=0,y=0,             # the coordinates of the node in GCS 
                                    # or grid coords
                                    # depending on use_grid
            label="node1",          # Human-readable node name
            restx=1,resty=1,        # restraints: 1 = restrained, 0 = free
            loadx=0,loady=0,        # applied loading (kips)
            use_grid=True           # whether to use grid for x,y
            )
    # (add more nodes)
    T2.add_node(x=1,y=0, label="node2", restx=1,resty=1, loadx=0,loady=0,    use_grid=True)

    T2.add_node(x=0,y=1, label="node3", restx=0,resty=0, loadx=10,loady=30,  use_grid=True)

    T2.add_material("matrl_1_3", # human-readable material name
                    29000    # E, youngs modulus, ksi
                )
    T2.add_material("matrl_2", 10000)


    A1_3 = 100 # in^2, cross-sectional area of elements
    A_2 = 50
    T2.add_elem(nodeStart       = "node1", # label of element start node
                nodeEnd        = "node2",  # label of element end node
                label          = "elem1",  # label of element
                area           = A1_3, #in^2  # cross-sectional area of element
                materialname   = "matrl_1_3"   # name of element's material
                )
    T2.add_elem(nodeStart       = "node2", # label of element start node
                nodeEnd        = "node3",  # label of element end node
                label          = "elem2",  # label of element
                area           = A_2, #in^2  # cross-sectional area of element
                materialname   = "matrl_2"   # name of element's material
                )
    T2.add_elem(nodeStart       = "node3", 
                nodeEnd        = "node1",  # label of element end node
                label          = "elem3",  # label of element
                area           = A1_3, #in^2  # cross-sectional area of element
                materialname   = "matrl_1_3"   # name of element's material
                )
    

    # Run analysis and get results

    ret2 = T2.run_analysis(debug=True) # debug = True will print all intermediate data. 
    plot_deformed_truss(ret2, exaggeration=100, force_visual_scale = .5, zoom_factor = .8) # exaggeration is a multiplier to make the displacements visually distinct

    """ 
    Finally, we will attempt to solve the third test truss, which is unstable.
    It will error.
    """
    T3 = TrussModel2D()
    T3.set_xgrid(3*12) # in
    T3.set_ygrid(6*12) # in
    T3.add_node(x=0,y=0,             # the coordinates of the node in GCS 
                                    # or grid coords
                                    # depending on use_grid
            label="node1",          # Human-readable node name
            restx=0,resty=1,        # restraints: 1 = restrained, 0 = free
            loadx=7,loady=0,        # applied loading (kips)
            use_grid=True           # whether to use grid for x,y
            )
    # (add more nodes)
    T3.add_node(x=2,y=0, label="node2", restx=1,resty=1, loadx=0,loady=0,    use_grid=True)

    T3.add_node(x=1,y=1, label="node3", restx=0,resty=0, loadx=-7,loady=0,  use_grid=True)


    T3.add_material("matrl_1", 10000)
    
    # this truss is unstable and should error

    A = 200 # in^2, cross-sectional area of elements
    T3.add_elem(nodeStart       = "node1", # label of element start node
                nodeEnd        = "node3",  # label of element end node
                label          = "elem1",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl_1"   # name of element's material
                )
    T3.add_elem(nodeStart       = "node2", # label of element start node
                nodeEnd        = "node3",  # label of element end node
                label          = "elem2",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl_1"   # name of element's material
                )
    
    

    # Run analysis and get results

    ret3 = T3.run_analysis(debug=True) # debug = True will print all intermediate data. 
    plot(ret3, exaggeration=100, force_visual_scale = .5, zoom_factor = .8) # exaggeration is a multiplier to make the displacements visually distinct



if __name__ == "__main__":
    raise SystemExit(main())
    
