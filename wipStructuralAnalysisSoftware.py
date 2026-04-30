import numpy as np
import math


# no OOP

def check_stable(S_ff):
    """
    Checks the stability of the structure by attempting to perform a Cholesky decomposition
    on the free-free stiffness matrix S_ff. If S_ff is not positive definite, it
    raises a ValueError indicating that the structure is unstable.
    """
    try:
        # np.linalg.cholesky(S_ff)
        # detect anything that is not positive semidefinite
        p = np.linalg.eigvals(S_ff)
        print(f"Eigenvalues of S_ff: {p}")
        
        
        # determine the rank
        s = np.linalg.svdvals(S_ff)
        rank = np.count_nonzero(s > 1e-10)
        print(f"Rank of S_ff: {rank}")
        # if np.any(p <= 0):
        #     raise ValueError("Structure Is SemiDefinite")
        #det = np.linalg.det(S_ff)
        #print(f"Determinant of S_ff: {det}")

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
    # a) determine the num elemes in the structure
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

def solve_displacements(S_ff, app_loads_free, fixed_distributed_loads_free):
    # solve
    equiv_load_free = app_loads_free - fixed_distributed_loads_free
    
    displacements_free = np.linalg.solve(S_ff, equiv_load_free)
    
    #displacements_free = np.linalg.pinv(S_ff, rcond=None, hermitian=True,rtol=1*10**-12) @ equiv_load_free
    return displacements_free

def solve_reactions(S_rf, displacements_free, fixed_distributed_loads_restr, loads_restr = 0):
    """
    Solves for the reactions at the restrained degrees of freedom based on the 
    free displacements and the fixed distributed load equivalent vector.

    Parameters:
    S_rf: The stiffness matrix with the restrained degrees of 
        freedom on the rows and the free degrees of freedom on the columns
    displacements_free: A 1D array of the displacements corresponding to the free degrees of freedom,
        obtained from solving the system of equations for the free degrees of freedom.
    fixed_distributed_loads_restr: A 1D array of the equivalent nodal loads 
        from distributed loads corresponding to the restrained degrees of freedom, 
        after accounting for the pin releases, in the global coordinate system.
    Returns:
    reactions_restr: A 1D array of the reactions at the restrained degrees of freedom
    """
    # solve for reactions
    reactions_restr = S_rf @ displacements_free + fixed_distributed_loads_restr - loads_restr
    return reactions_restr



def run_analysis(nodes, elem, elast, areas, inertia, restr, pins, app_loads, w, p):

    elem -= 1 # adjust for 0 indexing. This gets done once. When elem is passed into the other functions,
              # the object would get modified again inside each function if we subtract 1 from elem in each function.
              # To avoid this, we will just subtract 1 from elem once here at the start of the analysis.

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
    np.set_printoptions(precision=3)
    print(S_ff)
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
    displacements_free = solve_displacements(S_ff, app_loads_free, fixed_distributed_loads_free)
    # solve for reactions
    # $$R_{support} = K_{rf} D_f + Q_{rf} - P_r$$
    reactions_restr = solve_reactions(S_rf, displacements_free, fixed_distributed_loads_restr, app_loads_restr)
    print("Free Displacements:")
    print(displacements_free)
    print("Restrained Reactions:")
    print(reactions_restr)

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

def frame1():

    nodes = np.array([  [ 0, 0], # Node 1 X Y
                        [ 2, 4],
                        [ 6, 4],
                        [12, 0]])
    
    elem = np.array([   [1,2],
                        [2,3],
                        [3,4]])
    
    elast = np.array([  [101*10**6],
                        [101*10**6],
                        [101*10**6]])
    
    areas = np.array([  [200],
                        [200],
                        [200]])
    
    inertia = np.array([[5*10**6],
                        [5*10**6],
                        [5*10**6]])
    
    restr_by_node = np.array([[1,1,0], 
                              [0,0,0],
                              [0,0,0],
                              [1,1,0]])
    
    restr = nodal_to_dof(restr_by_node)

    pins = np.array([   [0,0], # elem 1 start pinned=1, elem 1 end
                        [0,0],
                        [0,0]])
    
    app_loads_by_node = np.array([  [ 0, 0, 0], #Node 1 Load X, Y, Theta
                                    [10, 0, 0], 
                                    [ 0, 0, 0], 
                                    [ 0, 0, 0]])
    
    app_loads = nodal_to_dof(app_loads_by_node)
    
    w = np.array([  [0,0], # elem 1 intens_s, intens_e
                    [10/1000,10/1000],
                    [0,0]])
    
    #p is the axial load intensity per member
    p = np.array([[0],
                  [0],
                  [0]])
    
    run_analysis(nodes = nodes, 
                    elem = elem, 
                    elast = elast, 
                    areas = areas, 
                    inertia = inertia, 
                    restr = restr, 
                    pins = pins, 
                    app_loads = app_loads, 
                    w = w,
                    p = p
                )
    
def ex_6_7():
    nodes = np.array([  [0, 0], # Node 1 X Y, m
                        [9, 0],
                        [0, 6],
                        [9, 6],
                        [0, 12]])
    # nodes *= 1000 # convert to mm
    
    elem = np.array([   [1,3], # elem 1 start node idx, elem 1 end node idx, 1 based idx
                        [2,4],
                        [3,5],
                        [3,4],
                        [4,5]])
    
    # elast is 30 GPa for all members 
    elast = np.array([  [30*10**6], # in kN/mm^2, convert from GPa
                        [30*10**6],
                        [30*10**6],
                        [30*10**6],
                        [30*10**6]])
    
    # area is 75000 mm^2 for all members
    areas = np.array([  [75000], # in mm^2
                        [75000],
                        [75000],
                        [75000],
                        [75000]])
    areas = areas * 1/(1000*1000) # convert to mm^2
    
    # inertia is 4.8E8 mm^4 for all members
    inertia = np.array([ [4.8*10**8], # in mm^4
                         [4.8*10**8],
                         [4.8*10**8],
                         [4.8*10**8],
                         [4.8*10**8]])
    inertia = inertia * 1/(1000*1000*1000*1000) # convert to mm^4
    
    restr_by_node = np.array([[1,1,1], # Node 1 Restr X, Y, Theta
                              [1,1,1],
                              [0,0,0],
                              [0,0,0],
                              [0,0,0]])

    restr = nodal_to_dof(restr_by_node)

    # no pins in this problem
    pins = np.array([   [0,0], # elem 1 start pinned=1, elem 1 end pinned=1
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0]])
    
    wind = 12 # kN/m, convert to kN/mm

    w = np.array([  [0,0], # elem 1 intens_s, intens_e
                    [0,0],
                    [0,0],
                    [0,0],
                    [wind,wind]])
    
    #p is the axial load intensity per member
    p = np.array([[0],
                  [0],
                  [0],
                  [0],
                  [0]])
    
    app_loads_by_node = np.array([  [ 0, 0, 0], #Node 1 Load X, Y, Theta
                                    [ 0, 0, 0],      
                                    [80, 0, 0],
                                    [ 0, 0, 0],
                                    [40, 0, 0]])
    
    app_loads = nodal_to_dof(app_loads_by_node)

    run_analysis(nodes = nodes,
                    elem = elem,
                    elast = elast,
                    areas = areas,
                    inertia = inertia,
                    restr = restr,
                    pins = pins,
                    app_loads = app_loads,
                    w = w,
                    p = p)

# def ex_7_1():
#     nodes = np.array([  [0, 0], # Node 1 X Y, m
#                         [0, 5],
#                         [2.5,5],
#                         [5, 5],
#                         [5, 0]])
    
#     nodes *= 1000 # convert to mm

#     elem = np.array([   [1,2], # elem 1 start node idx, elem 1 end node idx, 1 based idx
#                         [2,3],
#                         [4,3]])
    
#     # elast is 200 GPa for all members
#     elast = np.array([  [200*10**6], # in kN/mm^2, convert from GPa
#                         [200*10**6],    
#                         [200*10**6]])
    
#     # area is 6500 mm^2 for all members
#     areas = np.array([  [6500], # in mm^2
#                         [6500],
#                         [6500]])
    
#     # inertia is 150E6 mm^4 for all members
#     inertia = np.array([ [150*10**6], # in mm^4
#                          [150*10**6],
#                          [150*10**6]])
    
#     restr_by_node = np.array([[1,1,1], # Node 1 Restr X, Y, Theta
#                               []


def frame2_stable():

    nodes = np.array([
        [0, 0],   # Node 1
        [0, 4],   # Node 2
        [4, 4]    # Node 3
    ])
    
    elem = np.array([
        [1, 2],
        [2, 3]
    ])
    
    elast = np.array([
        [101e6],
        [101e6]
    ])
    
    areas = np.array([
        [200],
        [200]
    ])
    
    inertia = np.array([
        [5e6],
        [5e6]
    ])
    
    # Node 1 fully fixed → stabilizes frame
    restr_by_node = np.array([
        [1,1,0],
        [0,0,0],
        [1,1,0]
    ])
    
    restr = nodal_to_dof(restr_by_node)

    # Fully rigid connections
    pins = np.array([
        [0,0],
        [1,0]
    ])
    
    app_loads_by_node = np.array([
        [0,0,0],
        [10,-10,0],
        [0,0,0]
    ])
    
    app_loads = nodal_to_dof(app_loads_by_node)
    
    w = np.array([
        [0,0],
        [0,0]
    ])
    
    p = np.array([
        [0],
        [0]
    ])
    
    run_analysis(nodes, elem, elast, areas, inertia, restr, pins, app_loads, w, p)



def frame2_unstable():

    nodes = np.array([
        [0, 0],   # Node 1
        [0, 4],   # Node 2
        [4, 4]    # Node 3
    ])
    
    elem = np.array([
        [1, 2],
        [2, 3]
    ])
    
    elast = np.array([
        [101e6],
        [101e6]
    ])
    
    areas = np.array([
        [200],
        [200]
    ])
    
    inertia = np.array([
        [5e6],
        [5e6]
    ])
    
    # Node 1 fully fixed → stabilizes frame
    restr_by_node = np.array([
        [1,1,0],
        [0,0,0],
        [0,0,0]
    ])
    
    restr = nodal_to_dof(restr_by_node)

    # Fully rigid connections
    pins = np.array([
        [0,0],
        [1,0]
    ])
    
    app_loads_by_node = np.array([
        [0,0,0],
        [10,-10,0],
        [0,0,0]
    ])
    
    app_loads = nodal_to_dof(app_loads_by_node)
    
    w = np.array([
        [0,0],
        [0,0]
    ])
    
    p = np.array([
        [0],
        [0]
    ])
    
    run_analysis(nodes, elem, elast, areas, inertia, restr, pins, app_loads, w, p)


if __name__ == "__main__":
    # run all the frame 2 cases.
    print("Running frame 2 stable case...")
    frame2_unstable()

    
    raise SystemExit()
