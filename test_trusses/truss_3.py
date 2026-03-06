# stiffness matrix is one folder up from this file
file = "../stiffness_matrix.py"
# run the stiffness matrix file to import the class and functions as st
# At top of hw_problems/2_6.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stiffness_matrix import TrussModel2D as st
from graph_truss import plot_deformed_truss as plot

def main():
    T3 = st()
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
    main()
