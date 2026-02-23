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

xgrid = 288
ygrid = 216
    

# define nodes in dict form
nodeList = []


# x: x (grid location if use_grid)
# y: y (grid loaction if use_grid)
# restx/y: restrained on x/y = 0 (false), = 1 (true)
# loadx/y: load in GCS in kips

def addnode(x,y,label,restx=0,resty=0,loadx=0,loady=0,use_grid=True):
    if use_grid:
        x *= xgrid
        y *= ygrid
    idx0 = len(nodeList)
    nodeList.append({
        "x":x,
        "y":y,
        "idx0":idx0,
        "label":label,
        "restx":restx,
        "resty":resty,
        "loadx":loadx,
        "loady":loady
        })


def findNodeIdxByLabel(label):
    for nodeData in nodeList:
        if nodeData["label"] == label:
            return nodeData["idx0"]

#--------------
# specify nodes
#--------------
#1
addnode(x=0,y=0,label="node1",restx=1,resty=1,loadx=0,loady=0,use_grid=True)
#2
addnode(x=1,y=0,label="node2",loady=-75)
#3
addnode(x=2,y=0,label="node3",resty=1)
addnode(x=3,y=0,label="node4",resty=1)
addnode(x=1,y=1,label="node5",loadx=25)
addnode(x=2,y=1,label="node6",loady=60)


nodes = np.array([[nodeList[idx]["x"], nodeList[idx]["y"]] 
                  for idx in range(len(nodeList))],dtype=float)
print("nodes = ")
print(nodes)




# returns a restr and applied loads matrix where the order is 
# [[x1][y1][x2][y2][x3]...] ordering the dof by node each getting 2 idxs

def parseNodesIntoDOFList(nodeList):
    restr = np.zeros((len(nodeList)*2,1),dtype=int)
    app_loads = np.zeros((len(nodeList)*2,1),dtype=float)
    
    idx = 0
    for nodeDict in nodeList:
        
        restr[idx*2] = nodeDict["restx"]
        restr[idx*2+1] = nodeDict["resty"]
        app_loads[idx*2] = nodeDict["loadx"]
        app_loads[idx*2+1] = nodeDict["loady"]
        
        idx += 1
        
    return restr, app_loads

restr, app_loads = parseNodesIntoDOFList(nodeList)

print("restr = ")
print(restr)
print("app_loads = ")
print(app_loads)



# Materials
materialDict = {
    0:{"name": "steel", # Material string, idx 1
       "youngMod": 29000 # E in [ksi]
       },
    1:{"name": "aluminum", # Material string, idx 0
       "youngMod": 10000 # E in [ksi]
       },
}

def getMaterialByName(name):
    for materialData in materialDict.values():
        if materialData["name"] == name:
            return materialData

def getMaterialEByName(name):
    matrl = getMaterialByName(name)
    return matrl["youngMod"]

elemList = []

# nodeStart if string, node label, if int, node idx0 for the start node
# nodeEnd   if string, node label, if int, node idx0 for the end   node
# area      cross section area in in^2
# Material  material name

def add_elem(nodeStart,nodeEnd,label,area=1,materialname="steel"):

    snodeidx = 0
    if isinstance(nodeStart, str):
        snodeidx = findNodeIdxByLabel(nodeStart)
    else:
        snodeidx = nodeStart
        
    enodeidx = 0
    if isinstance(nodeEnd, str):
        enodeidx = findNodeIdxByLabel(nodeEnd)
    else:
        enodeidx = nodeEnd
            
    idx0 = len(elemList)
    
    elemList.append({
        "start":snodeidx,
        "end":enodeidx,
        "label":label,
        "idx0":idx0,
        "area":area,
        "material_name":materialname
    })
    

#--------------
# specify elements
#--------------

add_elem(nodeStart      = "node1",
         nodeEnd        = "node2",
         label          = "elem1",
         area           = 8, #in^2
         materialname   = "steel")

add_elem(nodeStart      = "node2",
         nodeEnd        = "node3",
         label          = "elem2",
         area           = 8,
         materialname   = "steel")

add_elem(nodeStart      = "node3",
         nodeEnd        = "node4",
         label          = "elem3",
         area           = 16,
         materialname   = "aluminum")

add_elem(nodeStart      = "node5",
         nodeEnd        = "node6",
         label          = "elem4",
         area           = 8,
         materialname   = "steel")

add_elem(nodeStart      = "node2",
         nodeEnd        = "node5",
         label          = "elem5",
         area           = 8,
         materialname   = "steel")

add_elem(nodeStart      = "node3",
         nodeEnd        = "node6",
         label          = "elem6",
         area           = 8,
         materialname   = "steel")

add_elem(nodeStart      = "node1",
         nodeEnd        = "node5",
         label          = "elem7",
         area           = 12,
         materialname   = "steel")

add_elem(nodeStart      = "node2",
         nodeEnd        = "node6",
         label          = "elem8",
         area           = 12,
         materialname   = "steel")

add_elem(nodeStart      = "node3",
         nodeEnd        = "node5",
         label          = "elem9",
         area           = 12,
         materialname   = "steel")

add_elem(nodeStart      = "node4",
         nodeEnd        = "node6",
         label          = "elem10",
         area           = 16,
         materialname   = "aluminum")

elem = np.array([[elemData["start"],elemData["end"]] for elemData in 
                 elemList],dtype=int)
elast = np.array([[getMaterialEByName(elemData["material_name"])] 
                  for elemData in elemList])
areas = np.array([[elemData["area"]] for elemData in elemList])

print("elem = ")
print(elem)


print("elast = ")
print(elast)


# Areas

# def get_areas(arealist):
#     return np.array([[arealist[idx]] for idx in range(len(arealist))])

# areas = get_areas([8,8,16,8,8,8,12,12,12,16]) # areas per bar in [in^2]

print("areas = ")
print(areas)

# def get_local_truss_stiffness(E,A,L):
#     # base local stiffness matrix assumes only axial forces
#     k_local_base = np.array([[1,0,-1,0],
#                             [0,0,0,0],
#                             [-1,0,1,0],
#                             [0,0,0,0]])

#     # local stiffness matrix
#     k_local = (E*A/L)*k_local_base
#     return k_local

   
# def truss_local_stiffness(nodes, elem, elast, areas):
#     # a) determine the num elemes in the structure
#     num_elem = elem.shape[0]
#     print("There are "+str(num_elem)+" bars")
#     # b) reduce the idxs of the elems by 1? - not needed in my case, already done
#     k = [] # list of local stiffness matrs
#     # d) loop the elems to calculate member len
#     for m in range(num_elem):
#         start_node = elem[m,0]
#         end_node = elem[m,1]
#         X_start = nodes[start_node, 0]
#         Y_start = nodes[start_node, 1]
#         X_end = nodes[end_node, 0]
#         Y_end = nodes[end_node, 1]
#         dX = X_end - X_start
#         dY = Y_end - Y_start
#         L = np.sqrt(dX**2 + dY**2)
#         E = elast[m]
#         A = areas[m]
#         # c) init the local stiff matrix
#         # e) calculate the member local stiffness matricies
#         k.append(get_local_truss_stiffness(E, A, L))
        
#     return k

# k = truss_local_stiffness(nodes, elem, elast, areas)
    


## element_global_stiffness 
## Params:
    # nodes: 
        # nodes array: [[x,y]...]
    # elem:
        # element array: [[node_idx_start, node_idx_end]...]
    # elast:
        # Youngs mod vector: [[elem_1_elast]...]
    # areas: 
        # areas vector: [[elem_1_area]...]
        
## returns: 
    #K:
        # per-element list of global stiffness matricies
        # [[]]
        
def element_global_stiffness(nodes, elem, elast, areas):

    # a) determine the num elemes in the structure

    num_elem = elem.shape[0]
    print("There are "+str(num_elem)+" bars")
    K = [] # list of global stiffness matrs

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
        
        K.append(K_m)
        
        # we now have a list of member stiffness matricies K in 
        # global coords. Next, we must assemble the final global
        # stiffness matrix.
        
        
        
    return K

K = element_global_stiffness(nodes, elem, elast, areas)



print("K(1) = ")
print(K[0])
print("K(5) = ")
print(K[4])
print("K(10) = ")
print(K[9])




## assemble_global_stiffness
## Finds pairs of 
## Params: 
    # nodes: 
        # nodes array: [[x,y]...]
    # elem:
        # element array: [[node_idx_start, node_idx_end]...]
    # element_global_stiffness_list:
        # list of global-coordinate element stiffness matricies
        



def assemble_and_partition(K_members, restrained_dofs, elem):
    ##
    ## Part A: assemble the full structural stiffness matrix
    ##
    
    # this gets the largest node idx from the element list
    # a clever way to extract the number of nodes without
    # actually passing the node list
    num_nodes = int(np.max(elem)) + 1
    
    ndof = 2* num_nodes
    
    # make ndofxndof matrix for the main stiffness matrix
    
    S = np.zeros((ndof,ndof))
    
    # extract the number of elements by the length of the 
    # element list
    
    num_elem = len(elem)
    # print(num_elem)
    
    # next, for each element of S, we need to find the 
    # Stiffness by adding all matching dofs from the
    # individual stiffness matricies.
    
    # First, make lookups of the dof for each elem.
    
    # for m in range(num_elem):
    #     i = elem[m,0] # first node idx of the element (start)
    #     j = elem[m,1] # end node idx of the element
        
    #     # determine the dof list idxs for each member
    #     dof_ix = 2*i
    #     dof_iy = 2*i + 1
    #     dof_jx = 2*j
    #     dof_jy = 2*j + 1
        
    #     member_dofs = [dof_ix, dof_iy, dof jx, dof_jy]
        
    #     # Assemble S
    
    for Sx in range(ndof):
        for Sy in range(ndof):
            # Sx, Sy define the dof location that we need to sum.
            # we will iterate all the elements and find
            pass
        
#assemble_and_partition(K, restr, elem)