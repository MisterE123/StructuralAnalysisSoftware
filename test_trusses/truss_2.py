# stiffness matrix is one folder up from this file
file = "../stiffness_matrix.py"
# run the stiffness matrix file to import the class and functions as st
# At top of hw_problems/2_6.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stiffness_matrix import TrussModel2D as st
from graph_truss import plot_deformed_truss as plot

def main():
    T2 = st()
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
    plot(ret2, exaggeration=100, force_visual_scale = .5, zoom_factor = .8) # exaggeration is a multiplier to make the displacements visually distinct

if __name__ == "__main__":
    main()
