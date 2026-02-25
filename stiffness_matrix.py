# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 17:50:12 2026

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


"""
import numpy as np
import math

def element_global_stiffness(nodes, elem, elast, areas):

    # a) determine the num elemes in the structure

    num_elem = elem.shape[0]
    print("There are "+str(num_elem)+" bars")
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
    num_nodes = int(np.max(elem)) + 1
    
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
    print(num_nodes)
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
    u : elem-based local deformation vector.
    
    E : Youngs mod.
    
    A : cross-sectional area.
    
    L : Member original len.

    Returns
    -------
    N : Normal Force.

    """
    
    N = (u[2]-u[0])*E*A/L
    
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
        u = u_vector[idx]
        E = elast[idx]
        A = areas[idx]
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
        print("T_list = ")
        print(T_list)
        print("K(1) = ")
        print(K_list[0])
        print("K(5) = ")
        print(K_list[4])
        print("K(10) = ")
        print(K_list[9])
        
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
        print(d_f)
    
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
        displacements, forces = m.run_analysis(
            debug=True   # whether to print out all intermediate 
                         # calculations
                       )
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
        return displacements_by_node, forces_by_node, normal_forces
   

     
def main():
    
    """
        Running the file by itself will run the following code
        
        This solves the example truss
        
        Passing debug = True like we do will print all intermediate data
        
    """
    m = TrussModel2D()
    
    #--------------
    # specify grid
    #--------------
    m.set_xgrid(288)
    m.set_ygrid(216)
    
    #--------------
    # specify nodes
    #--------------
    
    #1
    m.add_node(x=0,y=0,label="node1",restx=1,resty=1,loadx=0,loady=0,use_grid=True)
    #2
    m.add_node(x=1,y=0,label="node2",loady=-75)
    #3
    m.add_node(x=2,y=0,label="node3",resty=1)
    m.add_node(x=3,y=0,label="node4",resty=1)
    m.add_node(x=1,y=1,label="node5",loadx=25)
    m.add_node(x=2,y=1,label="node6",loady=60)
    
    #--------------
    # Specify Materials
    #--------------
    
    m.add_material("steel", 29000)
    m.add_material("aluminum", 10000)

        
    #--------------
    # specify elements
    #--------------
    
    m.add_elem(nodeStart    = "node1",
             nodeEnd        = "node2",
             label          = "elem1",
             area           = 8, #in^2
             materialname   = "steel")
    
    m.add_elem(nodeStart      = "node2",
             nodeEnd        = "node3",
             label          = "elem2",
             area           = 8,
             materialname   = "steel")
    
    m.add_elem(nodeStart    = "node3",
             nodeEnd        = "node4",
             label          = "elem3",
             area           = 16,
             materialname   = "aluminum")
    
    m.add_elem(nodeStart    = "node5",
             nodeEnd        = "node6",
             label          = "elem4",
             area           = 8,
             materialname   = "steel")
    
    m.add_elem(nodeStart    = "node2",
             nodeEnd        = "node5",
             label          = "elem5",
             area           = 8,
             materialname   = "steel")
    
    m.add_elem(nodeStart    = "node3",
             nodeEnd        = "node6",
             label          = "elem6",
             area           = 8,
             materialname   = "steel")
    
    m.add_elem(nodeStart    = "node1",
             nodeEnd        = "node5",
             label          = "elem7",
             area           = 12,
             materialname   = "steel")
    
    m.add_elem(nodeStart    = "node2",
             nodeEnd        = "node6",
             label          = "elem8",
             area           = 12,
             materialname   = "steel")
    
    m.add_elem(nodeStart    = "node3",
             nodeEnd        = "node5",
             label          = "elem9",
             area           = 12,
             materialname   = "steel")
    
    m.add_elem(nodeStart    = "node4",
             nodeEnd        = "node6",
             label          = "elem10",
             area           = 16,
             materialname   = "aluminum")
    
    m.run_analysis(debug=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    
