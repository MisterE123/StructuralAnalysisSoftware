# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:40:15 2026

@author: gbrru
"""
import numpy as np
from matplotlib import pyplot as plt


def beam_local_stiffness(nodes, elem, elast, inertia):
    """    
    Calculates the individual per-element stiffness matricies
    which will be assembled into the full stiffness matrix.
    Local matricies need not be transformed since local maps to 
    global directly.
    
    Parameters
    ----------
    nodes : numpy (nnode × 2) matrix of node coordinates [X, Y ].
    
    elem : (nelem × 2) member connectivity [startNode, endNode].
    
    elast : (nelem × 1) elastic modulus per member.
    
    inertia : (nelem × 1) second moment of area per member.


    Returns
    -------
    Ks : (normal) list of numpy stiffness matricies.

    """
    num_elem = elem.shape[0]
    print("num_elem: " , num_elem)
    elem=elem-1
    Ks=[]
    for m in range(num_elem):
        start_node = elem[m,0]
        end_node = elem[m,1]
        
        X_start = nodes[start_node, 0]
        Y_start = nodes[start_node, 1]
        X_end = nodes[end_node,0]
        Y_end = nodes[end_node,1]
        
        dX = X_end - X_start
        dY = Y_end - Y_start
        
        L = np.sqrt(dX**2 + dY**2)
        
        E = elast[m]
        I = inertia[m]
        
        k_sub = np.array([
            [12,    6*L,     -12,    6*L],
            [6*L,   4*L**2,  -6*L,   2*L**2],
            [-12,   -6*L,    12,     -6*L],
            [6*L,   2*L**2,  -6*L,   4*L**2]
        ])
        
        k = E * (I /(L**3)) * k_sub
        
        Ks.append(k)
        
    return Ks

def beam_fixed_end_forces(nodes, elem, w):
    """
    

    Parameters
    ----------
    nodes : TYPE
        DESCRIPTION.
    elem : TYPE
        DESCRIPTION.
    w : TYPE
        DESCRIPTION.

    Returns
    -------
    Q_F : TYPE
        DESCRIPTION.

    """
    num_elem = elem.shape[0]
    elem=elem-1
    Q_F = []
    for m in range(num_elem):
        
        start_node = elem[m,0]
        end_node   = elem[m,1]
        
        X_start = nodes[start_node, 0]
        Y_start = nodes[start_node, 1]
        
        X_end   = nodes[end_node  , 0]
        Y_end   = nodes[end_node  , 1]
        
        dX = X_end - X_start
        dY = Y_end - Y_start
        
        L = np.sqrt(dX**2 + dY**2)
        
        wb = w[m,0]
        we = w[m,1]
        
        # positive slope or flat?
        flat = wb
        tri  = we - wb
        if wb > we:
            # its negative slope
            # get the flat part which is we
            flat = we
            tri  = wb - we
            
        qflat = flat*L*0.5*np.array([[1],
                                    [L/6],
                                    [1],
                                    [-L/6]])
        qtri = tri*L*(1/20)*np.array([[3],
                                      [2*L/3],
                                      [7],
                                      [-L]])
        if wb > we:
            
            # its negative slope

            qtri = tri*L*(1/20)*np.array([[7],
                                          [L],
                                          [3],
                                          [-2*L/3]])
        q = qflat + qtri
        
        Q_F.append(q)
    
    return Q_F


def get_free_and_restr_idxs(restrained_dofs):
    """
    

    Parameters
    ----------
    restrained_dofs : TYPE
        DESCRIPTION.

    Returns
    -------
    free_dof_idxs : TYPE
        DESCRIPTION.
    restr_dof_idxs : TYPE
        DESCRIPTION.

    """
    free_dof_idxs = []
    restr_dof_idxs = []
    for dof_idx in range(len(restrained_dofs)):
        if restrained_dofs[dof_idx][0] == 1:
            restr_dof_idxs.append(dof_idx)
        else:
            free_dof_idxs.append(dof_idx)
    return free_dof_idxs, restr_dof_idxs



def assemble_and_partition(K_members, Q_F, restrained_dofs, elem, app_loads):
    """
    

    Parameters
    ----------
    K_members : TYPE
        DESCRIPTION.
    Q_F : TYPE
        DESCRIPTION.
    restrained_dofs : TYPE
        DESCRIPTION.
    elem : TYPE
        DESCRIPTION.
    app_loads : TYPE
        DESCRIPTION.

    Returns
    -------
    S : TYPE
        DESCRIPTION.
    S_ff : TYPE
        DESCRIPTION.
    S_rf : TYPE
        DESCRIPTION.
    Peq : TYPE
        DESCRIPTION.
    Peqfree : TYPE
        DESCRIPTION.
    Peqrestr : TYPE
        DESCRIPTION.
    Q_fixed : TYPE
        DESCRIPTION.

    """
    elem = elem - 1
    ##
    ## Part A: assemble the full structural stiffness matrix
    ##
    
    # this gets the largest node idx from the element list
    # a clever way to extract the number of nodes without
    # actually passing the node list
    num_nodes = int(np.max(elem)) + 1
    
    ndof = 2 * num_nodes
    
    # num_unrestr = count_unrestrained_dof(restrained_dofs)

    # print("Sanity check: ", ndof == len(restrained_dofs))

    # make ndof x ndof matrix for the main stiffness matrix
    
    S = np.zeros((ndof,ndof))
    
    # also assemble the applied forces on the nodes q_fixed
    Q_fixed = np.zeros((ndof,1))
    
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
            
            # add in the equivalent fixed reaction from the distributed load
            # to Q_F
            #print(member_dofs[h])
            Q_fixed[member_dofs[h]] += Q_F[m][h][0]
            
            # assemble the stiffness matrix
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
    
    # delete free rows leaving restr
    temp_arr_2 = np.delete(S, free_dof_idxs, axis=0)  
    # delete restrained cols leaving free
    S_rf = np.delete(temp_arr_2, restr_dof_idxs, axis=1) 
    # we now have S, S_ff, and S_rf
        
    # we now also have Peq the distributed load
    # time to restrict it to Peqfree and Peqrestr
    Peq = app_loads-Q_fixed

    Peqfree  = np.delete(Peq,     restr_dof_idxs, axis=0)
    Peqrestr = np.delete(Q_fixed, free_dof_idxs,  axis=0)
    
    return S, S_ff, S_rf, Peq, Peqfree, Peqrestr, Q_fixed

def get_free_displacements(Sff, Peqf):
    """
    

    Parameters
    ----------
    Sff : TYPE
        DESCRIPTION.
    Peqf : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    
    df = (np.linalg.inv(Sff))@Peqf

    return df

def get_support_rxns(S_rf,df,Peqr):
    """
    
    Parameters
    ----------
    S_rf : TYPE
        DESCRIPTION.
    df : TYPE
        DESCRIPTION.
    Peqr : TYPE
        DESCRIPTION.

    Returns
    -------
    Pr : TYPE
        DESCRIPTION.

    """
    Pr = S_rf@df + Peqr
    return Pr




def make_report(nodes, elem, df, restr, scale, Pr):
    elem = elem - 1
    # make a report that shown inputs, outputs, and uses the shape
    # functions to show the deformed shape
    fig, ax = plt.subplots() # create a figure containing a single Axes.
    
    # First, plot the original shape and problem.
    # plot the nodes
    
    for nidx in range(len(nodes)):
        ax.plot(nodes[nidx][0],nodes[nidx][1],"s",color='grey')
        ax.text(nodes[nidx][0],nodes[nidx][1], "  "+str(nidx+1), color = "darkslategrey")
        
    for eidx in range(len(elem)):
        el = elem[eidx]
        el_node_start_x = nodes[el[0]][0]
        el_node_start_y = nodes[el[0]][1]

        el_node_end_x = nodes[el[1]][0]
        el_node_end_y = nodes[el[1]][1]
        
        elem_x_vals = np.linspace(el_node_start_x,el_node_end_x,100)
        elem_y_vals = np.linspace(el_node_start_y,el_node_end_y,100)
        
        ax.plot(elem_x_vals,elem_y_vals, color = "grey")
        
    
    # graph the deformed shape
    restr_idx = 0
    for nidx in range(len(nodes)):
        rot = 0
        nx = nodes[nidx][0]
        ny = nodes[nidx][1]
        if restr[nidx*2][0]==0:
            # disp is unrestrained
            ny += df[restr_idx]
        for dof in [nidx*2,nidx*2+1]:
            if restr[dof][0]==0:
                # it is unrestrained
                rot += df[restr_idx]
                
                
                
            
    plt.show()
    
    
    
    pass

def run_analysis(nodes,elem,elast,inertia,restr,app_loads,w,scale, debug):
    """
    Runs Direct Stiffness Analysis of a beam. Performs analysis then
    graphs and prints results

    Parameters
    ----------
    nodes : numpy (nnode × 2) matrix of node coordinates [X, Y ].
    
    elem : (nelem × 2) member connectivity [startNode, endNode].
    
    elast : (nelem × 1) elastic modulus per member.
    
    inertia : (nelem × 1) second moment of area per member.
    
    restr : (ndof × 1) restraint indicator vector (1 = restrained, 0 = free).
    
    app_loads : (ndof × 1) applied nodal load vector (forces and/or moments at 
            DOFs; set reactions to zero).
    
    w : (nelem × 2) matrix of distributed load intensities [w_b, w_e]
            where the first column gives the load intensity at the beginning 
            of the element and the second column gives the load intensity at 
            the end of the element.
            
    scale : scalar value that amplifies the deformations so they can be 
            visually seen.
    
    debug: print debug info.
    
    Returns
    -------
    None.

    """
    # Compute the element stiffness matrices
    K_members = beam_local_stiffness(nodes, elem, elast, inertia)
    # Compute the equivalent nodal load vectors for distributed loads
    Q_F = beam_fixed_end_forces(nodes, elem, w)
    # Assemble and partition
    S, S_ff, S_rf, Peq, Peqf, Peqr, Q_fixed = assemble_and_partition(K_members, 
                                                            Q_F, 
                                                            restr, 
                                                            elem, 
                                                            app_loads)
    df = get_free_displacements(S_ff, Peqf)
    
    Pr = get_support_rxns(S_rf, df, Peqr)
    
    make_report(nodes,elem,df,restr,scale,Pr)
    
    if debug:
        
        np.set_printoptions(precision=3)
        def pretty_print(name,array):
            print("\n")
            print(name+": \n")
            print(array)
            print("\n")

        for i in range(len(K_members)):
            k = K_members[i]
            pretty_print("K of elem "+str(i+1),k)
            pretty_print("Q_f"+str(i+1), Q_F[i])
            
        pretty_print("S",      S)
        pretty_print("S_ff",   S_ff)
        pretty_print("S_rf",   S_rf)
        pretty_print("Peq" ,   Peq)
        pretty_print("Peqf",   Peqf)
        pretty_print("Peqr",   Peqr)
        pretty_print("Qfixed", Q_fixed)
        pretty_print("Df",     df)
        pretty_print("Pr",     Pr)
    
    
def beam1():
    """
    Performs analysis of beam 1

    Returns
    -------
    None.

    """
    # [x_pos,y_pos (always 0)]
    nodes = np.array([[0,0],  # n1
                      [3,0]]) # n2
    
    # [node_beginning, node_end]
    elem = np.array([[1,2]]) # elem1
    
    # length must be elem [Young's modulus]
    elast = np.array([[200]]) # elem 1 elast, GPa
    
    inertia = np.array([[1*10**9]]) # elem 1 inertia, mm^4
    
    # per dof restr
    restr = np.array([[1], # node 1 vert restr
                      [1], # node 1 angular restr
                      [0], # node 2 vert restr
                      [0]])# node 2 angular restr
    # per dof loads
    app_loads = np.array([
        [ 0],  # kN
        [ 0],  # kN*mm
        [-4], # kN
        [ 0]   # kN*mm
        ])
    
    w = np.array([[0,0],
                  [0,0]])
    
    print("Running analysis of Beam 1...\n")
    run_analysis(nodes, elem, elast, inertia, restr, app_loads, w, 5, True)
    

def beam2():
    """
    Performs analysis of beam 2    

    Returns
    -------
    None.

    """
    
    pass
def beam3():
    """
    Performs analysis of beam 3

    Returns
    -------
    None.

    """
    pass

def beam_example():
    # [x_pos,y_pos (always 0)]
    nodes = np.array([[0  ,0],   # n1
                      [48 ,0],   # n2
                      [120,0]])  # n3
    
    # [node_beginning, node_end]
    elem = np.array([[1,2],     # elem1 node connectivity
                     [2,3]])    # elem2 node connectivity
    
    # length must be elem [Young's modulus]
    elast = np.array([[29000],  # elem 1 elast, ksi
                      [10000]]) # elem 2 elast, ksi
    
    inertia = np.array([[60],   # elem 1 inertia, in^4
                        [50]])  # elem 2 inertia, in^4
    
    # per dof restr
    restr = np.array([[1],      # node 1, dof 1 vert restr
                      [0],      # node 1, dof 2 angular restr
                      [0],      # node 2, dof 3 vert restr
                      [0],      # node 2, dof 4 ang.
                      [1],      # node 3, dof 5 vert
                      [0]])     # node 3, dof 6 angular restr
    
    # per dof loads
    app_loads = np.array([
                        [ 0],   # 1 # kip
                        [ 0],   # 2 # kip*in
                        [ 0],   # 3 # kip
                        [36],   # 4 # kip*in
                        [ 0],   # 5 # kip
                        [ 0]])  # 6 # kip*in
        
    
    w = np.array([[0.25,0.25],  # elem1 kip/in, s, e
                  [0   ,0   ]]) # elem2 kip/in, s, e
    
    print("Running analysis of Beam Example...\n")
    run_analysis(nodes, elem, elast, inertia, restr, app_loads, w, 5, True)
    

def main():
    """
    is called when the file is called directly

    Returns
    -------
    None.

    """
    beam_example()

if __name__ == "__main__":
    raise SystemExit(main())
