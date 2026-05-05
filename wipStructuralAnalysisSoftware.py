import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

np.set_printoptions(linewidth=np.inf)
# no OOP

def check_stable(S_ff):
    """
    Checks the stability of the structure by attempting to perform a Cholesky decomposition
    on the free-free stiffness matrix S_ff. If S_ff is not positive definite, it
    raises a ValueError indicating that the structure is unstable.
    """
    pass
    try:
        np.linalg.cholesky(S_ff)

    except np.linalg.LinAlgError:
        raise ValueError("Structure Is Unstable")
        

def dXdYL(nodes, elem):
    """
    Computes the differences in x and y coordinates (dX and dY) for each element in the structure.
    Parameters:
    nodes: A 2D array where each row corresponds to a node and contains its (x, y) coordinates.
    elem: A 2D array where each row corresponds to an element and contains the indices of its start and end nodes.
    Returns:
    Three lists: dX_list, dY_list, and L_list, containing the differences in x and y coordinates and the lengths for each element.
    """
    dX_list = []
    dY_list = []
    L_list = []
    for m in range(elem.shape[0]):
        start_node = elem[m,0]
        end_node = elem[m,1]
        X_start = nodes[start_node, 0]
        Y_start = nodes[start_node, 1]
        X_end = nodes[end_node, 0]
        Y_end = nodes[end_node, 1]
        dX = X_end - X_start
        dY = Y_end - Y_start
        L = np.sqrt(dX**2 + dY**2)
        dX_list.append(dX)
        dY_list.append(dY)
        L_list.append(L)

    return dX_list, dY_list, L_list


def element_transformation(nodes, elem, dX_list, dY_list, L_list):
    """
    Computes the transformation matrix for each element in the structure.
    Parameters:
    nodes: A 2D array where each row corresponds to a node and contains its (x, y) coordinates.
    elem: A 2D array where each row corresponds to an element and contains the indices of its start and end nodes.
    Returns:
    A list of transformation matrices, one for each element.
    """
    T_list = []
    for m in range(elem.shape[0]):

        dX = dX_list[m]
        dY = dY_list[m]
        L = L_list[m]

        c = dX/L # cos (theta)
        s = dY/L # sin (theta)
        
        T = np.array([[ c, s, 0, 0, 0, 0],
                      [-s, c, 0, 0, 0, 0],
                      [ 0, 0, 1, 0, 0, 0],
                      [ 0, 0, 0, c, s, 0],
                      [ 0, 0, 0,-s, c, 0],
                      [ 0, 0, 0, 0, 0, 1]]) 
        T_list.append(T)
    
    return T_list

def element_global_stiffness(elem, elast, areas, inertia, pins, T_list, L_list):
    # a) determine the num elems in the structure
    num_elem = elem.shape[0]
    K_list = [] # list of global stiffness matrs
    for m in range(num_elem):
        
        L = L_list[m]
        E = elast[m][0]
        A = areas[m][0]
        I = inertia[m][0]
        start_pin = pins[m,0]
        end_pin = pins[m,1]
        
        k_local = np.zeros((6,6))
        EA_L = E*A/L
        EI_L3 = E*I/(L**3)
        EI_L2 = E*I/(L**2)
        EI_L = E*I/L

        match (start_pin, end_pin):
            case (0, 0): # both ends fixed, MT = 0
                k_local = np.array([
                    [    EA_L,        0,         0,      -EA_L,         0,         0],
                    [       0, 12*EI_L3,   6*EI_L2,          0, -12*EI_L3,   6*EI_L2],
                    [       0,   6*EI_L2,   4*EI_L,          0,  -6*EI_L2,    2*EI_L],
                    [   -EA_L,        0,         0,       EA_L,         0,         0],
                    [       0,-12*EI_L3,  -6*EI_L2,          0,  12*EI_L3,  -6*EI_L2],
                    [       0,  6*EI_L2,   2*EI_L,           0,  -6*EI_L2,    4*EI_L] 
                ])
            
            case (1, 0): # start pinned, end fixed, MT = 1
                k_local = np.array([
                    [    EA_L,        0,         0,      -EA_L,         0,         0],
                    [       0,  3*EI_L3,         0,          0,  -3*EI_L3,   3*EI_L2],
                    [       0,        0,         0,          0,         0,         0],
                    [   -EA_L,        0,         0,       EA_L,         0,         0],
                    [       0, -3*EI_L3,         0,          0,   3*EI_L3,  -3*EI_L2],
                    [       0,  3*EI_L2,         0,          0,  -3*EI_L2,    3*EI_L]
                ])

            case (0, 1): # start fixed, end pinned, MT = 2
                k_local = np.array([
                    [    EA_L,        0,         0,      -EA_L,         0,         0],
                    [       0,  3*EI_L3,   3*EI_L2,          0,  -3*EI_L3,         0],
                    [       0,   3*EI_L2,   3*EI_L,          0,  -3*EI_L2,         0],
                    [   -EA_L,        0,         0,       EA_L,         0,         0],
                    [       0, -3*EI_L3,  -3*EI_L2,          0,   3*EI_L3,         0],
                    [       0,        0,         0,          0,         0,         0]
                ])

            case (1, 1): # both ends pinned, MT = 3
                k_local = np.array([
                    [    EA_L,        0,         0,      -EA_L,         0,         0],
                    [       0,        0,         0,          0,         0,         0],
                    [       0,        0,         0,          0,         0,         0],
                    [   -EA_L,        0,         0,       EA_L,         0,         0],
                    [       0,        0,         0,          0,         0,         0],
                    [       0,        0,         0,          0,         0,         0]
                ])
        

        # T is a 6x6 transformation matrix that transforms local element stiffness matrices to global coordinates.
        T = T_list[m] 
        TT = T.T
        print("k_local ",m,": ",k_local)
        K_m = TT @ k_local @ T #local transformed to global
        
        K_list.append(K_m)
        
        # we now have a list of member stiffness matricies K in 
        # global coords. Next, we must assemble the final global
        # stiffness matrix (see assemble_and_partition)
    return K_list 

def get_free_and_restr_idxs(restrained_dofs):
    free_dof_idxs = []
    restr_dof_idxs = []
    print(restrained_dofs)
    for dof_idx in range((restrained_dofs.shape[0])):

        if restrained_dofs[dof_idx][0] == 1:
            restr_dof_idxs.append(dof_idx)
        else:
            free_dof_idxs.append(dof_idx)
        print(free_dof_idxs)
    return free_dof_idxs, restr_dof_idxs

def get_member_dofs(i, j):
    """
    Given the start and end node indices (i and j) of an element, 
    this function returns a list of the corresponding degree of freedom (DOF) 
    indices for that element in the global stiffness matrix. 
    Each node has three DOFs: translation in x, translation in y, and rotation (theta). 
    The DOF indices are calculated based on the node indices and the assumption that 
    each node contributes three DOFs to the global stiffness matrix.
    Parameters:
    i: The index of the start node of the element.
    j: The index of the end node of the element.
    Returns:
    A list of DOF indices corresponding to the start and end nodes of the element in the global stiffness matrix.
    """
    dof_ix = 3*i          # corresponds to the 0th idx in K
    dof_iy = 3*i + 1      # corresponds to the 1st idx in K
    dof_i_theta = 3*i + 2 # corresponds to the 2nd idx in K
    dof_jx = 3*j          # corresponds to the 3rd idx in K
    dof_jy = 3*j + 1      # corresponds to the 3rd idx in K
    dof_j_theta = 3*j + 2 # corresponds to the 2nd idx in K
    # dof lookup
    return [dof_ix, dof_iy, dof_i_theta, dof_jx, dof_jy, dof_j_theta]

def assemble_and_partition_stiffness(K_members, restr, elem):

    # this gets the largest node idx from the element list
    # a clever way to extract the number of nodes without
    # actually passing the node list
    num_nodes = np.max(elem) + 1 # add 1 to adjust for 0 indexing
    ndof = 3 * num_nodes
    # make ndof x ndof matrix for the main stiffness matrix
    S = np.zeros((ndof,ndof))
    # extract the number of elements by the length of the 
    # element list
    for m in range(elem.shape[0]):
        i = elem[m,0] # first node idx of the element (start)
        j = elem[m,1] # end node idx of the element
        K = K_members[m] # collect the member's Stiffness matrix
        # determine the dof list idxs for each member
        member_dofs = get_member_dofs(i, j)
        
        # our approach will be to add each member matrix into the main
        # stiffness matrix
        
        # h and k take on the idx of K, and we collect the idxs of
        # S from the member_dofs list
        for h in range(len(member_dofs)):
            for k in range(len(member_dofs)):
                Sy = member_dofs[h]
                Sx = member_dofs[k]
                
                S[Sy][Sx] += K[h][k]
                
    # now we have a Full S stiffness matrix
    # print(S)
    
    # next we need to remove unrestrained DOFs to get S_ff
    # first copy S into S_ff and S_rf

    # get indexes of free and restrained DOFs for the delete operations
    free_dof_idxs, restr_dof_idxs = get_free_and_restr_idxs(restr)
    # delete restrained rows leaving free
    temp_arr = np.delete(S, restr_dof_idxs, axis=0) 
    # delete restrained cols leaving free
    S_ff = np.delete(temp_arr, restr_dof_idxs, axis=1) 
    # delete free rows leaving restr
    print(np.shape(S))
    temp_arr_2 = np.delete(S, free_dof_idxs, axis=0) 
    # delete restrained cols leaving free
    S_rf = np.delete(temp_arr_2, restr_dof_idxs, axis=1) 
    # we now have S, S_ff, and S_rf
    print("S_ff:" , S_ff)
    print("S_rf:",S_rf)
    return S, S_ff, S_rf

def get_equivalent_nodal_loads_list(w, p, elem, L_list, T_list, pins):
    """
    Computes the equivalent nodal loads for each element in the structure based on the distributed loads and their transformations through the pins (moment releases).
    Parameters:
    w: A 2D array where each row corresponds to an element and contains the distributed load intensities at the start and end of the element.
    p: A 2D array where each row corresponds to an element and contains the axial
         load intensity for that element.
    elem: A 2D array where each row corresponds to an element and contains the indices of its start and end nodes.
    L_list: A list containing the lengths of each element.
    T_list: A list of transformation matrices for each element, used to transform local equivalent nodal loads to global coordinates.
    pins: A 2D array where each row corresponds to an element and contains the pin conditions at the start and end of the element (0 for fixed, 1 for pinned).
    Returns:
    A list of GLOBAL equivalent nodal load vectors for each element, resulting from the distributed loads and their moment releases through the pins.
    """
    Qf_list = []
    for m in range(elem.shape[0]):
        f = np.zeros((6,1)) # fixed end equivalent nodal loads in local coords from the distributed load.
        # [[FAb],
        #  [FSb],
        #  [FMb],
        #  [FAe],
        #  [FSe],
        #  [FMe]]
        Qf = np.zeros((6,1)) # the final equivalent nodal loads in local coords after modifications based on moment release

        w_start = w[m,0]
        w_end = w[m,1]
        p_intens = p[m]
        L = L_list[m]
        T = T_list[m]

        # positive slope or flat?
        flat = w_start
        tri  = w_end - w_start

        if w_start > w_end:
            # its negative slope
            # get the flat part which is we
            flat = w_end
            tri  = w_start - w_end
            
        qflat = flat*L*0.5*np.array([   [0],
                                        [1],
                                        [L/6],
                                        [0],
                                        [1],
                                        [-L/6]])
        
        qtri = tri*L*(1/20)*np.array([  [0],
                                        [3],
                                        [2*L/3],
                                        [0],
                                        [7],
                                        [-L]])
        
        if w_start > w_end:
            
            # its negative slope

            qtri = tri*L*(1/20)*np.array([  [0],
                                            [7],
                                            [L],
                                            [0],
                                            [3],
                                            [-2*L/3]])
            
        f += qflat + qtri

        # add the axial load contribution

        p_flat = p_intens * L * 0.5 * np.array([[1],
                                                [0],    
                                                [0],
                                                [1],
                                                [0],
                                                [0]])
        f += p_flat
        # now we have the equivalent nodal loads in local coords, we must transform to global
        # Now f contains
        # [[FAb],
        #  [FSb],
        #  [FMb],
        #  [FAe],
        #  [FSe],
        #  [FMe]]
        # Now, depending on the pin conditions, we will apply the modifications to the 
        # forces based on the solved system when moments are released.
        start_pin = pins[m,0]
        end_pin = pins[m,1]
        # for ease of reading, we will assign each fixed end force to a named variable
        FAb = f[0][0]
        FSb = f[1][0]
        FMb = f[2][0]
        FAe = f[3][0]
        FSe = f[4][0]
        FMe = f[5][0]
        match (start_pin, end_pin):
            case (0, 0): # both ends fixed, MT = 0
                # no modifications needed
                Qf = f
            case (1, 0): # start pinned, end fixed, MT = 1
                Qf[0][0] = FAb
                Qf[1][0] = FSb - (3/(2*L))*FMb
                Qf[2][0] = 0
                Qf[3][0] = FAe
                Qf[4][0] = FSe + 3/(2*L)*FMb
                Qf[5][0] = FMe - 0.5*FMb
            case (0, 1): # start fixed, end pinned, MT = 2
                Qf[0][0] = FAb
                Qf[1][0] = FSb - 3/(2*L)*FMe
                Qf[2][0] = FMb - 0.5*FMe
                Qf[3][0] = FAe
                Qf[4][0] = FSe + (3/(2*L))*FMe
                Qf[5][0] = 0
            case (1, 1): # both ends pinned, MT = 3
                Qf[0][0] = FAb
                Qf[1][0] = FSb - (1/L)*(FMb+FMe)
                Qf[2][0] = 0
                Qf[3][0] = FAe
                Qf[4][0] = FSe + (1/L)*(FMb+FMe)
                Qf[5][0] = 0
        # we have finally applied the distributed loads through the pins
        Qf_global = T.T @ Qf
        Qf_list.append(Qf_global)
    return Qf_list

def assemble_and_partition_loads(app_loads, Qf_list, restr, elem):
    """
    Assembles the applied loads and the equivalent nodal loads from distributed loads, 
    and partitions them into free and restrained components based on the provided 
    restrictions.
    Parameters:
        app_loads: A 1D array of applied loads corresponding to each degree of freedom 
            in the global stiffness matrix.
        Qf_list: A list of equivalent nodal load vectors for each element, resulting 
            from the distributed loads and their transformations through the pins.
        restr: A 1D array indicating which degrees of freedom are restrained (1) and which
            are free (0).
        elem: A 2D array where each row corresponds to an element and contains the indices of
            its start and end nodes. This is used to determine the member DOFs for applying 
            the equivalent nodal loads.
    Returns:
        app_loads_free - A 1D array of applied loads corresponding to the free degrees
            of freedom, after accounting for the equivalent nodal loads from distributed loads.
        fixed_distributed_loads_free - A 1D array of the equivalent nodal loads from 
            distributed loads corresponding to the free degrees of freedom, after accounting for the pin releases
        fixed_distributed_loads_restr - A 1D array of the equivalent nodal loads from 
            distributed loads corresponding to the restrained degrees of freedom, after accounting for the pin releases
    """

    ndof = app_loads.shape[0]
    free_dof_idxs, restr_dof_idxs = get_free_and_restr_idxs(restr)

    app_loads_free = np.delete(app_loads, restr_dof_idxs, axis=0)
    app_loads_restr = np.delete(app_loads, free_dof_idxs, axis=0)
    print(f"restrained_dofs: {restr_dof_idxs}")
    print(f"app_loads_free: {app_loads_free}")
    print(f"app_loads: {app_loads}")

    # to build the fixed free load vector, we will start with an ndof vector, add the     
    # equivalent nodal loads for the distributed loads, and then delete the restrained dofs
    fixed_distributed_loads = np.zeros((ndof, 1))
    for m in range(elem.shape[0]):
        # the start and end node idx for the current element
        i = elem[m,0] # first node idx of the element (start)
        j = elem[m,1] # end node idx of the element

        # these are the fixed end forces in local coordinates for the current element, 
        # after being transferred though pin connections (moment releases)
        
        Qf = Qf_list[m]

        # FAb = Qf[0][0]
        # FSb = Qf[1][0]
        # FMb = Qf[2][0]
        # FAe = Qf[3][0]
        # FSe = Qf[4][0]
        # FMe = Qf[5][0]
        member_dofs = get_member_dofs(i, j)

        for h in range(len(member_dofs)):
            dof_idx = member_dofs[h]
            fixed_distributed_loads[dof_idx] += Qf[h][0]
    
    fixed_distributed_loads_free = np.delete(fixed_distributed_loads, restr_dof_idxs, axis=0)
    fixed_distributed_loads_restr = np.delete(fixed_distributed_loads, free_dof_idxs, axis=0)
    return app_loads_free, fixed_distributed_loads_free, fixed_distributed_loads_restr, app_loads_restr

def solve_displacements(S_ff, app_loads_free, fixed_distributed_loads_free, solve_psuedo = False):
    # solve
    equiv_load_free = app_loads_free - fixed_distributed_loads_free
    
    if solve_psuedo:
        displacements_free = np.linalg.pinv(S_ff, rcond=None, hermitian=True,rtol=1*10**-12) @ equiv_load_free
    else:
        displacements_free = np.linalg.solve(S_ff, equiv_load_free)
    return displacements_free

def solve_reactions(S_rf, displacements_free, fixed_distributed_loads_restr, loads_restr = 0):
    """
    Solves for the reactions at the restrained degrees of freedom based on the 
    free displacements and the fixed distributed load equivalent vector.

    Parameters:
    S_rf: The stiffness matrix with the restrained degrees of 
        freedom on the rows and the free degrees of freedom on the columns
    displacements_free: A 1D array of the displacements corresponding to the 
        free degrees of freedom, obtained from solving the system of equations 
        for the free degrees of freedom.
    fixed_distributed_loads_restr: A 1D array of the equivalent nodal loads 
        from distributed loads corresponding to the restrained degrees of freedom, 
        after accounting for the pin releases, in the global coordinate system.
    Returns:
    reactions_restr: A 1D array of the reactions at the restrained degrees of freedom
    """
    # solve for reactions
    reactions_restr = S_rf @ displacements_free + fixed_distributed_loads_restr - loads_restr
    return reactions_restr

def make_node(axe,x,y,rot,color,size,truss_node):
    """
    Draws a beam node with given parameters on a plot Axe
    Parameters
    ----------
    axe : Plot Axes.
    x : Plot x pos.
    y : Plot x pos.
    rot : node rotation [rad].
    color : node color.
    size : node size.
    truss_node : 1 draws as a circle, 0 as a square

    Returns
    -------
    None.

    """
    rot = (rot * (180/3.14159))-45
    sides = 4
    if truss_node == 1:
        sides = 20
    axe.plot(x,y,
             color = color,
             markersize = size,
             marker = (sides,0,rot))

def shape_N1(x,L):
    """
    Gives the unit deflection along the beam due to the first dof

    Parameters
    ----------
    x : linspace list along beam.
    L : length of beam

    Returns
    -------
    linspace of unit deflection along beam.
    """
  
    return (1-(3*((x)/L)**2)+(2*((x)/L)**3))
    
def shape_N2(x,L):
    return ((x)*(1-((x)/L))**2)

def shape_N3(x,L):
    return ((3*((x)/L)**2)-(2*((x)/L)**3))

def shape_N4(x,L):
    return ((((x)**2)/L)*(-1+(x)/L))


def make_report(nodes, elem, pins, original_pins, T_list, L_list, restr, 
                df, Pr, truss_nodes, scale=1, 
                units={"length":"[L]", "force" :"[F]"}):
    
    fig, deflect_dia = plt.subplots() # create a figure containing a single Axes.

    text_size = 5
    
    num_nodes = int(np.max(elem)) + 1
    ndof = 3 * num_nodes
    df = df * scale

    disp_data_titles = ["Node", "Vert Disp ["+units["length"]+"]", "Rot [rad]"]

    disp_table_data = [
        disp_data_titles,
    ]
    
    react_data_titles = ["Node","Reaction","Units"]
    
    react_table_data = [
        react_data_titles,
    ]
    # calculate plot bounds
    max_x = max(nodes[:,0])
    min_x = min(nodes[:,0])
    xrange = (max_x - min_x)
    x_pad = xrange/10
    label_offset = (xrange+2*x_pad)/100
    max_y = max(nodes[:,1])
    min_y = min(nodes[:,1])
    yrange = max_y-min_y
    y_pad = yrange/10

    for nidx in range(len(nodes)):
        make_node(deflect_dia,nodes[nidx][0],nodes[nidx][1],0,"grey",text_size,
                  truss_node=truss_nodes[nidx][0])
        deflect_dia.text(nodes[nidx][0]+2*label_offset,
                nodes[nidx][1]+2*label_offset, 
                str(nidx+1), 
                color = "grey", 
                size = text_size,
                weight = "bold",
                bbox = dict(boxstyle="circle", fc = "none", ec = "grey")
                )

    unde_beam_diagram = 0 # just declaring
    
    for eidx in range(len(elem)):
        el = elem[eidx]
        el_node_start_x = nodes[el[0]][0]
        el_node_start_y = nodes[el[0]][1]

        el_node_end_x = nodes[el[1]][0]
        el_node_end_y = nodes[el[1]][1]
        
        elem_x_vals = np.linspace(el_node_start_x,el_node_end_x,100)
        elem_y_vals = np.linspace(el_node_start_y,el_node_end_y,100)
        
        # plot the undeformed member
        unde_beam_diagram = deflect_dia.plot(elem_x_vals,elem_y_vals, 
                                             color = "grey")
        
        
        label_x = ((el_node_end_x+el_node_start_x)/2)+3*label_offset
        label_y = ((el_node_end_y+el_node_start_y)/2)+3*label_offset
        

        deflect_dia.text(label_x,label_y," "+str(eidx+1)+" ",
                color = "grey",
                size = text_size,
                weight = "bold",
                bbox = dict(boxstyle="square", fc = "none", ec = "grey"))
        
    # graph the deformed shape
    
    restr_idx = 0
    Pr_idx = 0
    dof_defl = np.zeros(ndof)
    dof_pos = np.zeros(ndof)
    
    # iterate through the node indicies
    
    for nidx in range(len(nodes)):
        rot = 0
        nx = nodes[nidx][0]
        ny = nodes[nidx][1]
        
        dfx = 0
        dfy = 0
        dfrot = 0

        # if the node is restrained on that dof then it will have a reaction
        # if not, it will have a deflection
        if restr[nidx*3][0]==0:
            # disp x is unrestrained
            dfx = df[restr_idx][0]
            nx += dfx
            restr_idx += 1
        else:
            # there will be a horiz force reaction
            react = (Pr[Pr_idx][0])
            react_table_data.append([str(nidx+1),f"RIGHT {react:.5g}", "["+units["force"]+"]"])
            Pr_idx += 1
        if restr[nidx*3+1][0]==0:
            # disp y is unrestrained
            dfy = df[restr_idx][0]
            ny += dfy
        else:
            # there will be a vertical force rxn
            react = (Pr[Pr_idx][0])
            react_table_data.append([str(nidx+1),f"UP {react:.5g}", "["+units["force"]+"]"])
            Pr_idx += 1
        if restr[nidx*3+2][0]==0:
            # rot is unrestrained
            dfrot = df[restr_idx][0]
            rot += dfrot
            restr_idx += 1
        else:
            # there will be a moment reaction
            react = (Pr[Pr_idx][0])
            react_table_data.append([str(nidx+1),f"CW {react:.5g}", "["+units["force"]+"⋅"+units["length"]+"]"])
            Pr_idx += 1
        make_node(deflect_dia,nx,ny,rot,"black",text_size,truss_node=truss_nodes[nidx][0])

        dof_defl[nidx*3] = dfx
        dof_defl[nidx*3+1] = dfy
        dof_defl[nidx*3+2]= dfrot
        
        dof_pos[nidx*3] = nx
        dof_pos[nidx*3+1] = ny
        dof_pos[nidx*3+2]= rot
        
        disp_table_data.append([str(nidx+1),f"{(dfy/scale):.5g}",f"{(dfrot/scale):.5g}"])
        
    for eidx in range(len(elem)):
        el = elem[eidx]
        
        
        el_node_start_x = nodes[el[0]][0]
        el_node_start_y = nodes[el[0]][1]
        L = L_list[eidx]
        
        
        Ub=dof_defl[el[0]*3]
        Vb=dof_defl[el[0]*3+1]
        Rb=dof_defl[el[0]*3+2]
        Ue=dof_defl[el[1]*3]
        Ve=dof_defl[el[1]*3+1]
        Re=dof_defl[el[1]*3+2]
        
        T = T_list[eidx]
        T_2 = T[0:2,0:2]
        T_2T = T_2.T
        DEFl = np.array([Ub,Vb,Rb,Ue,Ve,Re])
        
        # local deflections
        ub,vb,rb,ue,ve,re = (T@DEFl).flatten()
        
        x = np.linspace(0,L,100)
        
        v_deflected = (shape_N1(x,L)*vb+shape_N2(x,L)*rb+
                 shape_N3(x,L)*ve+shape_N4(x,L)*re)
        
            
        # Axial deflection u(x) (linear interpolation)
        u_deflected = ub * (1 - x/L) + ue * (x/L)
        
        
        # this is a temporary measure... if the member is a truss-like, simply
        # draw it straight. This does not work for pinned-fixed and will just
        # draw 
        if original_pins[eidx][0] == 1 and original_pins[eidx][1] == 1:
            #u_deflected = x*0
            v_deflected = vb * (1 - x/L) + ve * (x/L)
            
        # 4. Transform Local Deflected Coordinates back to Global
        # True local coordinates are (x + u, v)
        local_coords = np.vstack((x + u_deflected, v_deflected))
        
        global_coords = T_2T@local_coords
        
        global_coords_x = global_coords[0] + el_node_start_x
        global_coords_y = global_coords[1] + el_node_start_y
        

        # transform x and the shape into Global
        
        
        de_beam_diagram = deflect_dia.plot(global_coords_x,global_coords_y, color = "black")
    
    deflect_dia.set_xbound(min_x-x_pad, max_x + x_pad)      
    deflect_dia.set_aspect(1, adjustable='datalim')
    deflect_dia.set_xlabel("["+units["length"]+"]")
    deflect_dia.set_ylabel("["+units["length"]+"]")
    plt.show()
    
    fig2, defl_chart = plt.subplots() # create a figure containing a single Axes.
    defl_chart.axis('off')  # hide axes
    table = defl_chart.table(cellText=disp_table_data, loc='center')
    table.scale(1, 1.5)
    plt.title("Displacements")
    plt.show()
    
    
    fig3, react_chart = plt.subplots()
    react_chart.axis('off')  # hide axes
    react_table = react_chart.table(cellText=react_table_data, loc='center')
    react_table.scale(1, 1.5)
    plt.title("Reactions")
    plt.show()



def weld_all_free_pins(nodes, elem, pins):
    """
    Welds free pins to the lowest-idx element attached to the node if all 
    attached elements are pinned. This is a common modeling approach for 
    trusses, where members are often modeled as having pinned connections at 
    their ends. However, if a node is connected to multiple elements and all 
    of those connections are pinned, it can lead to an 
    unrestrained degree of freedom for the rotation of that node. To address 
    this issue, this optional function identifies nodes that have pinned 
    connections at all attached elements and welds the pin to the lowest-
    indexed element, effectively making it a fixed connection and eliminating 
    the unrestrained rotational degree of freedom.

    Parameters:
    nodes: A 2D array where each row corresponds to a node and contains its 
        (x, y) coordinates.
    elem: A 2D array where each row corresponds to an element and contains the
        indices of its start and end nodes.
    pins: A 2D array where each row corresponds to an element and contains the
        pin conditions at the start and end of the element (0 for fixed, 1 
        for pinned).
    Returns:
    A modified version of the pins array where free pins have been welded to 
    the lowest-indexed element if all attached elements are pinned.
    """

    # trusses are often modeled as elems with pins at each end. Unfortunately, that leaves an
    # unrestrained DOF for the rotation of the node. This function finds nodes that have 
    # pinned connections at all attached elements, and welds the pin to the lowest-idx element.
    truss_nodes = np.zeros((nodes.shape[0],1))
    # truss_nodes is used to graph truss nodes as circles

    # iterate thru nodes,
    for n in range(nodes.shape[0]):
        first_connection_elem = -1 # this will get welded if is_all_pinned
        first_connection_elem_idx = -1 # this is 0 (elem start) or 1 (end)
        is_all_pinned = True # true if all elems connected at the end have a pin
        # find all connected elem
        for m in range(elem.shape[0]):
            for e in (0,1): # check start and beginning
                if elem[m][e] == n: # then elem is attached to the node at end "e"
                    if first_connection_elem == -1: # only set the first connected elem once
                        first_connection_elem = m
                        first_connection_elem_idx = e
                    if pins[m][e] == 0: # if it is not pinned, record that fact
                        is_all_pinned = False
        if is_all_pinned and first_connection_elem != -1: # then weld the first connected pin
            pins[first_connection_elem][first_connection_elem_idx] = 0 # make it not pinnned.
            truss_nodes[n][0] = 1
    return pins, truss_nodes
                

def run_analysis(nodes, elem, elast, areas, inertia, restr, pins, 
                 app_loads, w, p, weld_free_pins = False, solve_pseudo = False,
                 scale = 1, units={"length":"[L]", "force" :"[F]"}):

    
    elem -= 1 # adjust for 0 indexing. This gets done once. When elem is passed into the other functions,
              # the object would get modified again inside each function if we subtract 1 from elem in each function.
              # To avoid this, we will just subtract 1 from elem once here at the start of the analysis.

    # prepare trusses for analysis
    truss_nodes = np.zeros((nodes.shape[0],1))
    original_pins = np.copy(pins)
    if weld_free_pins:
        
        pins, truss_nodes = weld_all_free_pins(nodes, elem, pins)
        print("pins:")
        print(pins)

    # get the dX, dY, and L for each element which are common properties for the member
    # calculations, so we only want to compute them once.

    dX_list, dY_list, L_list = dXdYL(nodes, elem)
    
    # get the transformation matrix for each element, which is also used multiple times
    T_list = element_transformation(nodes, elem, dX_list, dY_list, L_list)
    # create the global stiffness matrix for each element in global coordinates.
    K_list = element_global_stiffness(elem, elast, areas, inertia, pins, T_list, L_list)
    print("T_list[0]:\n", T_list[0])
    print("K_list[0]:\n", K_list[0])
    S, S_ff, S_rf = assemble_and_partition_stiffness(K_list, restr, elem)
    print("S_ff:")
    np.set_printoptions(precision=5)
    print(S_ff)
    if not solve_pseudo:
        check_stable(S_ff)
    # get the equivalent nodal loads for the distributed loads
    Qf_list = get_equivalent_nodal_loads_list(w, p, elem, L_list, T_list, pins)
    # assemble and partition the loads
    app_loads_free, \
        fixed_distributed_loads_free, \
        fixed_distributed_loads_restr, \
        app_loads_restr = \
        assemble_and_partition_loads(app_loads, Qf_list, restr, elem)
    print("app_loads_free:\n", app_loads_free)
    print("app_loads_restr:\n", app_loads_restr)
    print("fixed_distributed_loads_free:\n", fixed_distributed_loads_free)
    print("fixed_distributed_loads_restr:\n", fixed_distributed_loads_restr)
    # solve
    displacements_free = solve_displacements(S_ff, app_loads_free, fixed_distributed_loads_free, solve_pseudo)
    # solve for reactions
    # $$R_{support} = K_{rf} D_f + Q_{rf} - P_r$$
    reactions_restr = solve_reactions(S_rf, displacements_free, fixed_distributed_loads_restr, app_loads_restr)
    print("Free Displacements:")
    print(displacements_free)
    print("Restrained Reactions:")
    print(reactions_restr)
    make_report(nodes=nodes,
                elem=elem,
                pins=pins,
                original_pins=original_pins,
                T_list=T_list, 
                L_list=L_list, 
                restr=restr, 
                df=displacements_free, 
                Pr=reactions_restr,
                scale=scale,
                truss_nodes=truss_nodes,
                units = units,
                )

def nodal_to_dof(by_node):
    """
    Convert node-based array to 
    degree-of-freedom-based array.
    Parameters:
    by_node: A 2D array where each row corresponds to a node and contains the restrictions for that node.
                [[val_x, val_y, val_theta], # node 1 vals
                    ...                     # more nodes
                ]
    Returns:
    A 1D array where each element corresponds to a degree of freedom
    """
    by_dof = np.zeros((3*(by_node.shape[0]), 1))

    for i in range(by_node.shape[0]):
        by_dof[i*3][0]   = by_node[i][0]
        by_dof[i*3+1][0] = by_node[i][1]
        by_dof[i*3+2][0] = by_node[i][2]
    return by_dof

def dof_to_nodal(by_dof):
    """
    Convert degree-of-freedom-based array to 
    node-based array.
    Parameters:
    by_dof: A 1D array where each element corresponds to a degree of freedom
    Returns:
    A 2D array where each row corresponds to a node and contains the restrictions for that node.
                [[val_x, val_y, val_theta], # node 1 vals
                    ...                     # more nodes
                ]
    """
    num_nodes = len(by_dof) // 3
    by_node = np.zeros((num_nodes, 3))

    for i in range(num_nodes):
        by_node[i][0] = by_dof[i*3][0]
        by_node[i][1] = by_dof[i*3+1][0]
        by_node[i][2] = by_dof[i*3+2][0]
    return by_node

def get_truss_interia(areas):
    # assumes a square cross section and returns a default MOI for trusses
    # or other structures for which one does not care about the inertia
    
    inertia = np.zeros((areas.shape[0],1))
    for aidx in range(areas.shape[0]):
        inertia[aidx][0] = ((areas[aidx][0])**2)/12
    return inertia

def get_all_pins(elem):
    # returns a pins array with all members pinned (moment released)
    return np.ones((elem.shape[0],2))

def get_elem_start_end_zeros(elem):
    return np.zeros((elem.shape[0],2))

def get_no_pins(elem):
    return get_elem_start_end_zeros(elem)


def get_elem_const_array(val,elem):
    """
    returns a per-element array of constant values (eg for areas or elast)

    Parameters
    ----------
    val : const.
    elem : the element connectivity matrix.

    Returns
    -------
    an n_elems x 1 numpy array.
    """
    return np.ones((elem.shape[0],1))*val

def Truss_1():
    
    print("Running analysis on Truss 1:")
    # all in meters, converted to mm
    nodes = np.array([
            [0,0],
            [4,0],
            [0,2],
            [-3,2]
        ])
    nodes *= 1000 #convert to mm
    
    elem = np.array([
            [1,2],
            [2,3],
            [3,4],
            [4,1],
            [1,3]
        ])
    
    elast = get_elem_const_array(200, elem) # kPa/mm^2
    areas = get_elem_const_array(100, elem) # mm^2
    inertia = get_truss_interia(areas) # mm^4, assumes square cross section
    pins = get_all_pins(elem) # makes all elements pinned
    
    restr_by_node = np.array([
            [1,1,0],
            [1,1,0],
            [0,0,0],
            [0,0,0]
        ])
    
    restr = nodal_to_dof(restr_by_node)
    
    app_loads_by_node = np.array([
            [0,0,0],
            [0,0,0],
            [0,-12,0],
            [7,-12,0]
        ])
    
    app_loads = nodal_to_dof(app_loads_by_node)
    
    # no distributed loads
    w = get_elem_start_end_zeros(elem)
    p = get_elem_const_array(0,elem)
    
    run_analysis(nodes = nodes,
                        elem = elem,
                        elast = elast,
                        areas = areas,
                        inertia = inertia,
                        restr = restr,
                        pins = pins,
                        weld_free_pins=True,
                        app_loads = app_loads,
                        w = w,
                        p = p,
                        scale = 10,
                        units = {"length":"mm", "force" :"kN"},
                    )
    
def Truss_2():
    print("Running analysis on Truss 2:")
    
    # Node Coordinates (Converted from ft to inches)
    # Node 1: (0, 0)
    # Node 2: (6, 0)
    # Node 3: (10, 0)
    # Node 4: (6, 4)
    nodes = np.array([
            [0, 0],
            [6, 0],
            [10, 0],
            [6, 4]
        ])
    nodes *= 12 # Convert ft to inches
    
    # Element Connectivity (using node indices 1-4)
    elem = np.array([
            [1, 2], # Element 1
            [2, 3], # Element 2
            [3, 4], # Element 3
            [4, 1], # Element 4
            [2, 4]  # Element 5
        ])
    
    # Properties
    elast = get_elem_const_array(10000, elem) # ksi
    areas = get_elem_const_array(0.25, elem)  # in^2
    inertia = get_truss_interia(areas) 
    pins = get_all_pins(elem) 
    
    # Restraints [UX, UY, RZ] (1 = fixed, 0 = free)
    # Node 1: Pin (Fixed X, Y)
    # Node 3: Roller (Fixed Y)
    # Node 4: Roller (Fixed Y) 
    restr_by_node = np.array([
            [1, 1, 0], # Node 1
            [0, 0, 0], # Node 2
            [0, 1, 0], # Node 3
            [0, 1, 0]  # Node 4
        ])
    restr = nodal_to_dof(restr_by_node)
    
    # Applied Loads [FX, FY, MZ] (Units in kips)
    # Node 2: 10 kip downward
    # Node 3: 8 kip right
    # Node 4: 6 kip left
    app_loads_by_node = np.array([
            [0, 0, 0],    # Node 1
            [0, -10, 0],  # Node 2
            [8, 0, 0],    # Node 3
            [-6, 0, 0]    # Node 4
        ])
    app_loads = nodal_to_dof(app_loads_by_node)
    
    # No distributed loads
    w = get_elem_start_end_zeros(elem)
    p = get_elem_const_array(0, elem)
    
    run_analysis(nodes = nodes,
                 elem = elem,
                 elast = elast,
                 areas = areas,
                 inertia = inertia,
                 restr = restr,
                 pins = pins,
                 weld_free_pins = True,
                 app_loads = app_loads,
                 w = w,
                 p = p,
                 scale = 25,
                 units = {"length": "in", "force": "kip"},
                )        
def Beam_1():
    print("Running analysis on Beam 1:")
    # Units: meters converted to mm
    nodes = np.array([
            [0, 0],
            [3, 0],
            [6, 0]
        ])
    nodes *= 1000 
    
    elem = np.array([
            [1, 2],
            [2, 3]
        ])
    
    # E = 200 GPa = 200 kN/mm^2, I = 4e6 mm^4
    elast = get_elem_const_array(200, elem)
    inertia = get_elem_const_array(4e6, elem)
    areas = get_elem_const_array(1000, elem) # Dummy area for beam analysis
    pins = get_no_pins(elem) # Fixed-fixed connections between elements
    
    # Restraints [UX, UY, RZ] (1=fixed)
    # Node 1: Pin, Node 2: Roller, Node 3: Roller
    restr_by_node = np.array([
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ])
    restr = nodal_to_dof(restr_by_node)
    
    # Applied Loads [FX, FY, MZ]
    # Node 2 has a 20 kN-m clockwise moment (-20)
    app_loads_by_node = np.array([
            [0, 0, 0],
            [0, 0, -20000], # kN*mm
            [0, 0, 0]
        ])
    app_loads = nodal_to_dof(app_loads_by_node)
    
    w = get_elem_start_end_zeros(elem)
    p = get_elem_const_array(0, elem)
    
    run_analysis(nodes=nodes, elem=elem, elast=elast, areas=areas, 
                 inertia=inertia, restr=restr, pins=pins, 
                 app_loads=app_loads, w=w, p=p, scale=30,
                 units={"length": "mm", "force": "kN"})
    
def Beam_2():
    print("Running analysis on Beam 2:")
    # Units: ft converted to inches
    nodes = np.array([
            [0, 0],
            [4, 0],
            [9, 0],
            [15, 0]
        ])
    nodes *= 12 
    
    elem = np.array([
            [1, 2],
            [2, 3],
            [3, 4]
        ])
    
    elast = get_elem_const_array(10000, elem)
    inertia = get_elem_const_array(50, elem)
    areas = get_elem_const_array(100, elem) 
    pins = get_no_pins(elem)
    
    # Node 1: Pin, Node 3: Roller
    restr_by_node = np.array([
            [1, 1, 0], # Node 1
            [0, 0, 0], # Node 2
            [0, 1, 0], # Node 3
            [0, 0, 0]  # Node 4
        ])
    restr = nodal_to_dof(restr_by_node)
    
    w= np.array([
            [0,0],
            [0,2.27],
            [2.27,5]
        ])
    w*=(-1/12)

    app_loads = nodal_to_dof(np.zeros((4, 3)))
    print(app_loads)
    p = get_elem_const_array(0, elem)
    
    run_analysis(nodes=nodes, elem=elem, elast=elast, areas=areas, 
                 inertia=inertia, restr=restr, pins=pins, 
                 app_loads=app_loads, w=w, p=p, scale=1,
                 units={"length": "in", "force": "kip"})
# def frame1():

#     nodes = np.array([  [ 0, 0], # Node 1 X Y
#                         [ 2, 4],
#                         [ 6, 4],
#                         [12, 0]])
    
#     elem = np.array([   [1,2],
#                         [2,3],
#                         [3,4]])
    
#     elast = np.array([  [101*10**6],
#                         [101*10**6],
#                         [101*10**6]])
    
#     areas = np.array([  [200],
#                         [200],
#                         [200]])
    
#     inertia = np.array([[5*10**6],
#                         [5*10**6],
#                         [5*10**6]])
    
#     restr_by_node = np.array([[1,1,0], 
#                               [0,0,0],
#                               [0,0,0],
#                               [1,1,0]])
    
#     restr = nodal_to_dof(restr_by_node)

#     pins = np.array([   [0,0], # elem 1 start pinned=1, elem 1 end
#                         [0,0],
#                         [0,0]])
    
#     app_loads_by_node = np.array([  [ 0, 0, 0], #Node 1 Load X, Y, Theta
#                                     [10, 0, 0], 
#                                     [ 0, 0, 0], 
#                                     [ 0, 0, 0]])
    
#     app_loads = nodal_to_dof(app_loads_by_node)
    
#     w = np.array([  [0,0], # elem 1 intens_s, intens_e
#                     [10/1000,10/1000],
#                     [0,0]])
    
#     #p is the axial load intensity per member
#     p = np.array([[0],
#                   [0],
#                   [0]])
    
#     run_analysis(nodes = nodes, 
#                     elem = elem, 
#                     elast = elast, 
#                     areas = areas, 
#                     inertia = inertia, 
#                     restr = restr, 
#                     pins = pins, 
#                     app_loads = app_loads, 
#                     w = w,
#                     p = p
#                 )
    
# def ex_6_7():
#     nodes = np.array([  [0, 0], # Node 1 X Y, m
#                         [9, 0],
#                         [0, 6],
#                         [9, 6],
#                         [0, 12]])
#     # nodes *= 1000 # convert to mm
    
#     elem = np.array([   [1,3], # elem 1 start node idx, elem 1 end node idx, 1 based idx
#                         [2,4],
#                         [3,5],
#                         [3,4],
#                         [4,5]])
    
#     # elast is 30 GPa for all members 
#     elast = np.array([  [30*10**6], # in kN/mm^2, convert from GPa
#                         [30*10**6],
#                         [30*10**6],
#                         [30*10**6],
#                         [30*10**6]])
    
#     # area is 75000 mm^2 for all members
#     areas = np.array([  [75000], # in mm^2
#                         [75000],
#                         [75000],
#                         [75000],
#                         [75000]])
#     areas = areas * 1/(1000*1000) # convert to mm^2
    
#     # inertia is 4.8E8 mm^4 for all members
#     inertia = np.array([ [4.8*10**8], # in mm^4
#                          [4.8*10**8],
#                          [4.8*10**8],
#                          [4.8*10**8],
#                          [4.8*10**8]])
#     inertia = inertia * 1/(1000*1000*1000*1000) # convert to mm^4
    
#     restr_by_node = np.array([[1,1,1], # Node 1 Restr X, Y, Theta
#                               [1,1,1],
#                               [0,0,0],
#                               [0,0,0],
#                               [0,0,0]])

#     restr = nodal_to_dof(restr_by_node)

#     # no pins in this problem
#     pins = np.array([   [0,1], # elem 1 start pinned=1, elem 1 end pinned=1
#                         [0,1],
#                         [1,1],
#                         [1,1],
#                         [1,1]])
    
#     wind = 12 # kN/m, convert to kN/mm

#     w = np.array([  [0,0], # elem 1 intens_s, intens_e
#                     [0,0],
#                     [0,0],
#                     [0,0],
#                     [wind,wind]])
    
#     #p is the axial load intensity per member
#     p = np.array([[0],
#                   [0],
#                   [0],
#                   [0],
#                   [0]])
    
#     app_loads_by_node = np.array([  [ 0, 0, 0], #Node 1 Load X, Y, Theta
#                                     [ 0, 0, 0],      
#                                     [80, 0, 0],
#                                     [ 0, 0, 0],
#                                     [40, 0, 0]])
    
#     app_loads = nodal_to_dof(app_loads_by_node)

#     run_analysis(nodes = nodes,
#                     elem = elem,
#                     elast = elast,
#                     areas = areas,
#                     inertia = inertia,
#                     restr = restr,
#                     pins = get_all_pins(elem),
#                     weld_free_pins=True,
#                     app_loads = app_loads,
#                     w = w,
#                     p = p,
#                     scale = 1,
#                     units = {"length":"mm",
#                              "force" :"kN"}
                    
#                     )

# # def ex_7_1():
# #     nodes = np.array([  [0, 0], # Node 1 X Y, m
# #                         [0, 5],
# #                         [2.5,5],
# #                         [5, 5],
# #                         [5, 0]])
    
# #     nodes *= 1000 # convert to mm

# #     elem = np.array([   [1,2], # elem 1 start node idx, elem 1 end node idx, 1 based idx
# #                         [2,3],
# #                         [4,3]])
    
# #     # elast is 200 GPa for all members
# #     elast = np.array([  [200*10**6], # in kN/mm^2, convert from GPa
# #                         [200*10**6],    
# #                         [200*10**6]])
    
# #     # area is 6500 mm^2 for all members
# #     areas = np.array([  [6500], # in mm^2
# #                         [6500],
# #                         [6500]])
    
# #     # inertia is 150E6 mm^4 for all members
# #     inertia = np.array([ [150*10**6], # in mm^4
# #                          [150*10**6],
# #                          [150*10**6]])
    
# #     restr_by_node = np.array([[1,1,1], # Node 1 Restr X, Y, Theta
# #                               []


# def frame2_stable():

#     nodes = np.array([
#         [0, 0],   # Node 1
#         [0, 4],   # Node 2
#         [4, 4]    # Node 3
#     ])
    
#     elem = np.array([
#         [1, 2],
#         [2, 3]
#     ])
    
#     elast = np.array([
#         [101e6],
#         [101e6]
#     ])
    
#     areas = np.array([
#         [200],
#         [200]
#     ])
    
#     inertia = np.array([
#         [5e6],
#         [5e6]
#     ])
    
#     # Node 1 fully fixed → stabilizes frame
#     restr_by_node = np.array([
#         [1,1,0],
#         [0,0,0],
#         [1,1,0]
#     ])
    
#     restr = nodal_to_dof(restr_by_node)

#     # Fully rigid connections
#     pins = np.array([
#         [0,0],
#         [1,0]
#     ])
    
#     app_loads_by_node = np.array([
#         [0,0,0],
#         [10,-10,0],
#         [0,0,0]
#     ])
    
#     app_loads = nodal_to_dof(app_loads_by_node)
    
#     w = np.array([
#         [0,0],
#         [0,0]
#     ])
    
#     p = np.array([
#         [0],
#         [0]
#     ])
    
#     run_analysis(nodes, elem, elast, areas, inertia, restr, pins, app_loads, w, p)



# def frame2_unstable():

#     nodes = np.array([
#         [0, 0],   # Node 1
#         [0, 4],   # Node 2
#         [4, 4]    # Node 3
#     ])
    
#     elem = np.array([
#         [1, 2],
#         [2, 3]
#     ])
    
#     elast = np.array([
#         [101e6],
#         [101e6]
#     ])
    
#     areas = np.array([
#         [200],
#         [200]
#     ])
    
#     inertia = np.array([
#         [5e6],
#         [5e6]
#     ])
    
#     # Node 1 fully fixed → stabilizes frame
#     restr_by_node = np.array([
#         [1,1,0],
#         [0,0,0],
#         [0,0,0]
#     ])
    
#     restr = nodal_to_dof(restr_by_node)

#     # Fully rigid connections
#     pins = np.array([
#         [0,0],
#         [1,0]
#     ])
    
#     app_loads_by_node = np.array([
#         [0,0,0],
#         [10,-10,0],
#         [0,0,0]
#     ])
    
#     app_loads = nodal_to_dof(app_loads_by_node)
    
#     w = np.array([
#         [0,0],
#         [0,0]
#     ])
    
#     p = np.array([
#         [0],
#         [0]
#     ])
    
#     run_analysis(nodes, elem, elast, areas, inertia, restr, pins, app_loads, w, p)

# def hw1():
#     nodes = np.array([
#         [0,0],
#         [72,125]
#     ])
    
#     elem = np.array([[1,2]])

#     elast = np.array([[29000]])

#     areas = np.array([[50]])

#     inertia = np.array([[200]])

#     restr_by_node = np.array([
#         [1,1,0],
#         [1,0,0]])
    
#     restr = nodal_to_dof(restr_by_node)

#     pins = np.array([[0,0]])

#     app_loads_by_node = np.array([
#         [0,0,30*12],
#         [0,-80,0]])
#     app_loads = nodal_to_dof(app_loads_by_node)
#     w = np.array([
#         [0,0]
#     ])
    
#     p = np.array([
#         [0]
#     ])

#     run_analysis(nodes, elem, elast, areas, inertia, restr, pins, app_loads, w, p)
    

    
if __name__ == "__main__":
    #Truss_1()
    #Truss_2()
    #Beam_1()
    Beam_2()
    raise SystemExit()
