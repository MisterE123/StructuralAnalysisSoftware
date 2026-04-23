# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:40:15 2026

@author: gbrru
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter



def beam_local_stiffness(nodes, elem, elast, inertia):
    """    
    Calculates the individual per-element stiffness matricies
    which will be assembled into the full stiffness matrix.
    Local matricies need not be transformed since local maps to 
    global directly.
    
    Parameters
    ----------
    nodes : numpy (nnode × 2) matrix of node coordinates [X, Y ].
    elem : numpy (nelem × 2) member connectivity [startNode, endNode].
    elast : numpy (nelem × 1) elastic modulus per member.
    inertia : numpy (nelem × 1) second moment of area per member.


    Returns
    -------
    Ks : list of numpy stiffness matricies.

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
    nodes : numpy (nnode × 2) matrix of node coordinates [X, Y ].
    elem : numpy (nelem × 2) member connectivity [startNode, endNode].
    w : numpy (nelem × 2) distributed loading. Positive loads are downwards
        [startLoadIntensity, endLoadIntensity], in [Force/Length]

    Returns
    -------
    Q_F : list of member numpy (4 × 1) element equivalent reactions from 
    applied loads. [[dof_start_applied_reaction], #[Force]
                    [dof_start_moment_reaction],  #[Force*Length]
                    [dof_end_applied_reaction],   #[Force]
                    [dof_end_moment_reaction]]    #[Force*Length]

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
    Returns free and restrained dofs as a list given the restrained dof
    flag vector.

    Parameters
    ----------
    restrained_dofs : (ndof × 1) restraint indicator vector 
    (1 = restrained, 0 = free).

    Returns
    -------
    free_dof_idxs : list of free dof indicies.
    restr_dof_idxs : list of restrained dof indicies.

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
    Creates the assembled stiffness matricies and Equivalent fixed reaction
    forces vector from the applied loads by assembling them from lists of the 
    members' stiffness matricies and Equivalent fixed reaction forces vectors

    Parameters
    ----------
    K_members : nelem length list of numpy (4 × 4) stiffness matricies
    Q_F : list of member numpy (4 × 1) element equivalent reactions from 
        applied loads. [[dof_start_applied_reaction], #[Force]
                        [dof_start_moment_reaction],  #[Force*Length]
                        [dof_end_applied_reaction],   #[Force]
                        [dof_end_moment_reaction]]    #[Force*Length]
    restrained_dofs : (ndof × 1) restraint indicator vector (1 = restrained, 0 = free).
    elem : numpy (nelem × 2) member connectivity [startNode, endNode].
    app_loads : (ndof × 1) applied nodal load vector (forces and/or moments at 
        DOFs; set reactions to zero).

    Returns
    -------
    S : Full numpy (ndof × ndof) Stiffness Matrix.
    S_ff : Free-Free (ndof_free × ndof_free) Stiffness Matrix.
    S_rf : Restrained (ndof_restrained × ndof_free) Stiffness Matrix.
    Peq : Equivalent loads (app_loads - Q_fixed).
    Peqfree : numpy (ndof_free × 1) free dof loads from Peq.
    Peqrestr : numpy (ndof_restr × 1) restrained dof loads from Peq.
    Q_fixed : full numpy (ndof × 1) fixed reaction-equivalent vector.
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
    Gets the displacements at free degrees of freedom

    Parameters
    ----------
    S_ff : Free-Free (ndof_free × ndof_free) Stiffness Matrix.

    Peqf : numpy (ndof_free × 1) free dof loads from Peq.

    Returns
    -------
    df : (ndof_free × 1) displacement vector.

    """
    
    df = (np.linalg.inv(Sff))@Peqf

    return df

def get_support_rxns(S_rf,df,Peqr):
    """
    Gets support reactions at restrained dofs
    
    Parameters
    ----------
    S_rf : Restrained (ndof_restrained × ndof_free) Stiffness Matrix.
    df : (ndof_free × 1) displacement vector.
    Peqr : numpy (ndof_restr × 1) restrained dof loads from Peq.

    Returns
    -------
    Pr : numpy (ndof_restr × 1) restrained dof reactions
# 
    """
    Pr = S_rf@df + Peqr
    return Pr

def make_node(axe,x,y,rot,color,size):
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

    Returns
    -------
    None.

    """
    rot = (rot * (180/3.14159))-45
    axe.plot(x,y,
             color = color,
             markersize = size,
             marker = (4,0,rot))

def shape_N1(x,x1,x2):
    """
    Gives the unit deflection along the beam due to the first dof

    Parameters
    ----------
    x : linspace list along beam.
    x1 : first value.
    x2 : end value.

    Returns
    -------
    linspace of unit deflection along beam.
    """
    L = x2-x1
    return (1-(3*((x-x1)/L)**2)+(2*((x-x1)/L)**3))
    
def shape_N2(x,x1,x2):
    L = x2-x1
    return ((x-x1)*(1-((x-x1)/L))**2)

def shape_N3(x,x1,x2):
    L = x2-x1
    return ((3*((x-x1)/L)**2)-(2*((x-x1)/L)**3))

def shape_N4(x,x1,x2):
    L = x2-x1
    return ((((x-x1)**2)/L)*(-1+(x-x1)/L))

def stability(m,r,j,h):
    """
    Returns the n of indeterminancy or -1 if unstable.

    Parameters
    ----------
    m : int, # members.
    r : int, # reactions (restrained dofs).
    j : int, # joints.
    h : int, # hinges.

    Returns
    -------
    int, -1 if unstable, 0 if determinant, >0 if indeterminant.
    """
    return max(-1,3*m+r-3*j-h)

def make_report(nodes, elem, df, restr, scale, Pr, units, title):    
    """
    Creates labeled deformed shape graph, displacement table, 
    and reactions table.

    Parameters
    ----------
    nodes : numpy (nnode × 2) matrix of node coordinates [X, Y ].
    elem : numpy (nelem × 2) member connectivity [startNode, endNode].
    df : numpy (ndof_free × 1) deflections .
    restr :(ndof × 1) restraint indicator vector (1 = restrained, 0 = free).
    scale : visual displacement scale multiplier (graph units remain accurate).
    Pr : numpy (ndof_restr × 1) restrained dof reactions
    units : unit dictionary: {"length":"mm",
                              "force" :"kN",
                              "press" :"GPa"}
        Pressure must be (Force / Length^2)
        Used for labels. 
    title : (str) Beam title.

    Returns
    -------
    None
    """
    # check stability
    #a roller is the same as a pin for beams, but should be given 2 reactions
    r = len(Pr)

    for yrest in restr[0:len(restr):2]:
        if yrest[0] == 1:
            r+=1
            
    m = len(elem)
    j = len(nodes)
    h = 0
    n = stability(m,r,j,h)
    #print("m: ",m," r: ",r," j: ",j," n: ", n)    
    
    
    elem = elem - 1
    num_nodes = int(np.max(elem)) + 1
    ndof = 2 * num_nodes
    df = df * scale
    
    # make a report that shown inputs, outputs, and uses the shape
    # functions to show the deformed shape
    fig, deflect_dia = plt.subplots() # create a figure containing a single Axes.

    # calculate the plot bounds
    max_x = max(nodes[:,...,0])
    min_x = min(nodes[:,...,0])
    xrange = (max_x - min_x)
    x_pad = xrange/10
    label_offset = (xrange+2*x_pad)/100
    #deflect_dia = axs[0]
    
    # First, plot the original shape and problem.
    # plot the nodes

    text_size = 5
    
    for nidx in range(len(nodes)):
        make_node(deflect_dia,nodes[nidx][0],nodes[nidx][1],0,"grey",text_size)
        deflect_dia.text(nodes[nidx][0]+2*label_offset,
                nodes[nidx][1]+2*label_offset, 
                str(nidx+1), 
                color = "grey", 
                size = text_size,
                weight = "bold",
                bbox = dict(boxstyle="circle", fc = "none", ec = "grey")
                )
        # TODO: check if restrained, draw appropriate restraints
        
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
        unde_beam_diagram = deflect_dia.plot(elem_x_vals,elem_y_vals, color = "grey")
        
        label_x = ((el_node_end_x+el_node_start_x)/2)
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
    
    disp_data_titles = ["Node", "Vert Disp ["+units["length"]+"]", "Rot [rad]"]

    disp_table_data = [
        disp_data_titles,
    ]
    
    react_data_titles = ["Node","Reaction","Units"]
    
    react_table_data = [
        react_data_titles,
    ]
    
    for nidx in range(len(nodes)):
        rot = 0
        nx = nodes[nidx][0]
        ny = nodes[nidx][1]
        
        dfy = 0
        dfrot = 0

        if restr[nidx*2][0]==0:
            # disp is unrestrained
            dfy = df[restr_idx][0]
            ny += dfy
            
            restr_idx += 1
        else:
            # there will be a force reaction
            react = (Pr[Pr_idx][0])
            react_table_data.append([str(nidx+1),f"{react:.3g}", "["+units["force"]+"]"])
            Pr_idx += 1
            
        if restr[nidx*2+1][0]==0:
            # rot is unrestrained
            dfrot = df[restr_idx][0]
            rot += dfrot
            restr_idx += 1
        else: 
            # there will be a moment reaction
            react = (Pr[Pr_idx][0])
            react_table_data.append([str(nidx+1),f"{react:.3g}", "["+units["force"]+"⋅"+units["length"]+"]"])
            Pr_idx += 1

        
        make_node(deflect_dia,nx,ny,rot,"black",text_size)

        dof_defl[nidx*2] = dfy
        dof_defl[nidx*2+1]=dfrot
        
        disp_table_data.append([str(nidx+1),f"{(dfy/scale):.3g}",f"{(dfrot/scale):.3g}"])
        
    
    de_beam_diagram = 0
    
    for eidx in range(len(elem)):
        el = elem[eidx]
        
        el_node_start_x = nodes[el[0]][0]
        el_node_start_y = nodes[el[0]][1]

        el_node_end_x = nodes[el[1]][0]
        el_node_end_y = nodes[el[1]][1]
        
        elem_x_vals = np.linspace(el_node_start_x,el_node_end_x,100)
        elem_y_vals = np.linspace(el_node_start_y,el_node_end_y,100)
        
        
        x = np.linspace(el_node_start_x,el_node_end_x,100)
        x1 = el_node_start_x
        x2 = el_node_end_x
        
        vb=dof_defl[el[0]*2]
        tb=dof_defl[el[0]*2+1]
        ve=dof_defl[el[1]*2]
        te=dof_defl[el[1]*2+1]
        
        shape = shape_N1(x,x1,x2)*vb+shape_N2(x,x1,x2)*tb+shape_N3(x,x1,x2)*ve+shape_N4(x,x1,x2)*te
        
        de_beam_diagram = deflect_dia.plot(x,shape, color = "black")
        

    deflect_dia.set_xbound(min_x-x_pad, max_x + x_pad)      
    deflect_dia.set_aspect(1, adjustable='datalim')    
    # deflect_dia.set_ybound(-y_range,y_range)

    deflect_dia.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{y/scale:.3g}")
    )
    
    deflect_dia.set_xlabel("["+units["length"]+"]")
    deflect_dia.set_ylabel("["+units["length"]+"]")
    
    deflect_dia.legend([unde_beam_diagram[0], de_beam_diagram[0]],
              ['Undeflected', 'Deflected ('+str(f"{(scale):.3g}")+'x)'])
    
    deformed_title = title + " Deformed Shape"
    if n==-1:
        deformed_title +="\n(UNSTABLE - DO NOT TRUST!)"
    plt.title(deformed_title)

    plt.show()
    
    # make the tables
    
    
    fig2, defl_chart = plt.subplots() # create a figure containing a single Axes.
    defl_chart.axis('off')  # hide axes
    table = defl_chart.table(cellText=disp_table_data, loc='center')
    table.scale(1, 1.5)
    plt.title(title + " Displacements")
    plt.show()
    
    
    fig3, react_chart = plt.subplots()
    react_chart.axis('off')  # hide axes
    react_table = react_chart.table(cellText=react_table_data, loc='center')
    react_table.scale(1, 1.5)
    plt.title(title + " Reactions")
    plt.show()

    
    
    
def run_analysis(nodes,elem,elast,inertia,restr,
                 app_loads,dist_loads,scale, units, debug, title):
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
    
    title : (str) Beam title.
    
    
    Returns
    -------
    None.

    """
    # Compute the element stiffness matrices
    K_members = beam_local_stiffness(nodes, elem, elast, inertia)
    # Compute the equivalent nodal load vectors for distributed loads
    Q_F = beam_fixed_end_forces(nodes, elem, dist_loads)
    # Assemble and partition
    S, S_ff, S_rf, Peq, Peqf, Peqr, Q_fixed = assemble_and_partition(K_members, 
                                                            Q_F, 
                                                            restr, 
                                                            elem, 
                                                            app_loads)
    df = get_free_displacements(S_ff, Peqf)
    
    Pr = get_support_rxns(S_rf, df, Peqr)
    
    make_report(nodes, elem, df, restr, scale, Pr, units, title)
    
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
                      [3000,0]]) # n2
    
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
        [-4],  # kN
        [ 0]   # kN*mm
        ])
    
    w1=8/1000
    w = np.array([[w1,w1],
                  [w1,w1]])
    
   
    print("Running analysis of Beam 1...\n")
    run_analysis(nodes, 
                 elem, 
                 elast, 
                 inertia, 
                 restr, 
                 app_loads, 
                 dist_loads = w, 
                 scale = 250, 
                 units = {"length":"mm",
                          "force" :"kN",
                          "press" :"GPa"}, 
                 debug = True,
                 title = "Beam 1")
    

def beam2():
    """
    Performs analysis of beam 2    

    Returns
    -------
    None.

    """
    nodes = np.array([[0,0],    # n1, in
                      [48,0],   # n2, in
                      [120,0],  # n3, in
                      [180,0]]) # n4, in
    
    # [node_beginning, node_end]
    elem = np.array([[1,2],  # elem1
                     [2,3],  # elem2
                     [3,4]]) # elem3
    
    # length must be elem [Young's modulus]
    elast = np.array([[29000],  # elem 1 elast, ksi
                      [10000],  # elem 2 elast, ksi
                      [29000]]) # elem 3 elast, ksi
    
    inertia = np.array([[80],  # elem 1 inertia, in^4
                        [70],  # elem 2 inertia, in^4
                        [60]]) # elem 3 inertia, in^4
    
    # per dof restr
    restr = np.array([[1], # node 1 vert restr
                      [0], # node 1 angular restr
                      [0], # node 2 vert restr
                      [0], # node 2 angular restr
                      [1], # node 3 vert restr
                      [0], # node 3 angular restr
                      [1], # node 4 vert restr
                      [0]])# node 4 angular restr
    
    # per dof loads
    app_loads = np.array([
        [ 0],  # kip
        [ 0],  # kip*in
        [ 0],  # kip
        [ 0],  # kip*in
        [ 0],  # kip
        [ 0],  # kip*in
        [ 0],  # kip
        [ 72]  # kip*in
    ])
    
    dist_loads = np.array([[0.25,0.75],  # elem 1, kip/in
                           [0.75,0],     # elem 2, kip/in
                           [0,0]])       # elem 3, kip/in
    
    print("Running analysis of Beam 2...\n")
    run_analysis(nodes, 
                 elem, 
                 elast, 
                 inertia, 
                 restr, 
                 app_loads, 
                 dist_loads, 
                 scale = 25,
                 units = {"length":"in",
                          "force" :"kip",
                          "press" :"ksi"}, 
                 debug = True,
                 title = "Beam 2")
    
def beam3():

    """
    Performs analysis of beam 3

    Returns
    -------
    None.

    """
     # [x_pos,y_pos (always 0)]
    nodes = np.array([[0,0],    # n1, in
                      [96,0],   # n2, in
                      [192,0]]) # n3, in
    
    # [node_beginning, node_end]
    elem = np.array([[1,2],  # elem1
                     [2,3]]) # elem2
    
    # length must be elem [Young's modulus]
    elast = np.array([[29000],  # elem 1 elast, ksi
                      [29000]]) # elem 2 elast, ksi
    
    inertia = np.array([[40],  # elem 1 inertia, in^4
                        [80]]) # elem 2 inertia, in^4
    
    # per dof restr
    restr = np.array([[0], # node 1 vert restr
                      [0], # node 1 angular restr
                      [1], # node 2 vert restr
                      [0], # node 2 angular restr
                      [0], # node 3 vert restr
                      [0]])# node 3 angular restr

    # per dof loads
    app_loads = np.array([
        [ -6], # kip
        [ 0],  # kip*in
        [ 0],  # kip
        [ 0],  # kip*in
        [ -6], # kip
        [ 0]]) # kip*in
    
    dist_loads = np.array([[0,0],  # elem 1, kip/in
                           [0,0]]) # elem 2, kip/in
    
    
    print("Running analysis of Beam 3...\n")
    run_analysis(nodes, 
                 elem, 
                 elast, 
                 inertia, 
                 restr, 
                 app_loads, 
                 dist_loads = dist_loads, 
                 scale = 5, 
                 units = {"length":"in",
                          "force" :"kip",
                          "press" :"ksi"},
                 debug = True,
                 title = "Beam 3")

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
        
    
    dist_loads = np.array([[0.25,0.25],  # elem1 kip/in, s, e
                           [0   ,0   ]]) # elem2 kip/in, s, e
    
    print("Running analysis of Beam Example...\n")
    run_analysis(nodes, 
                 elem, 
                 elast, 
                 inertia, 
                 restr, 
                 app_loads, 
                 dist_loads, 
                 scale = 25,
                 units = {"length":"in",
                          "force" :"kip",
                          "press" :"ksi"},
                 debug = True,
                 title = "Beam Example")
    

def main():
    """
    is called when the file is called directly

    Returns
    -------
    None.

    """
    beam1()
    beam2()
    beam3()

if __name__ == "__main__":
    raise SystemExit(main())
