# stiffness matrix is one folder up from this file
file = "../stiffness_matrix.py"
# run the stiffness matrix file to import the class and functions as st
# At top of hw_problems/2_6.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stiffness_matrix import TrussModel2D as st
from graph_truss import plot_deformed_truss as plot

def main():
    m = st()
    m.set_xgrid(8*1000) # mm
    m.set_ygrid(6*1000) # mm
    m.add_node(x=0,y=0,             # the coordinates of the node in GCS 
                                    # or grid coords
                                    # depending on use_grid
            label="node1",          # Human-readable node name
            restx=1,resty=1,        # restraints: 1 = restrained, 0 = free
            loadx=0,loady=0,        # applied loading
            use_grid=True           # whether to use grid for x,y
            )
    # (add more nodes)
    m.add_node(x=1,y=0, label="node2", restx=1,resty=1, loadx=0,loady=0,    use_grid=True)

    m.add_node(x=0,y=1, label="node3", restx=0,resty=0, loadx=0,loady=0,    use_grid=True)
    m.add_node(x=1,y=1, label="node4", restx=0,resty=0, loadx=6,loady=0,    use_grid=True)
    m.add_node(x=0,y=2, label="node5", restx=0,resty=0, loadx=0,loady=-20,  use_grid=True)
    m.add_node(x=1,y=2, label="node6", restx=0,resty=0, loadx=12,loady=-20, use_grid=True)

    m.add_material("matrl1", # human-readable material name
                    200    # E, youngs modulus, Gpa
                    )

    # (add more materials)

    A = 1000 # mm^2, cross-sectional area of elements
    m.add_elem(nodeStart       = "node1", # label of element start node
                nodeEnd        = "node3",  # label of element end node
                label          = "elem1",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    m.add_elem(nodeStart       = "node2", # label of element start node
                nodeEnd        = "node4",  # label of element end node
                label          = "elem2",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    m.add_elem(nodeStart       = "node3", 
                nodeEnd        = "node5",  # label of element end node
                label          = "elem3",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    m.add_elem(nodeStart       = "node4",
                nodeEnd        = "node6",  # label of element end node
                label          = "elem4",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    m.add_elem(nodeStart       = "node3",
                nodeEnd        = "node4",  # label of element end node
                label          = "elem5",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    m.add_elem(nodeStart       = "node5",
                nodeEnd        = "node6",  # label of element end node
                label          = "elem6",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    m.add_elem(nodeStart       = "node1",
                nodeEnd        = "node4",  # label of element end node
                label          = "elem7",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    m.add_elem(nodeStart       = "node2",
                nodeEnd        = "node3",  # label of element end node
                label          = "elem8",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    m.add_elem(nodeStart       = "node3",
                nodeEnd        = "node6",  # label of element end node
                label          = "elem9",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )
    m.add_elem(nodeStart       = "node4",
                nodeEnd        = "node5",  # label of element end node
                label          = "elem10",  # label of element
                area           = A, #in^2  # cross-sectional area of element
                materialname   = "matrl1"   # name of element's material
                )


     # Run analysis and get results

    ret = m.run_analysis(debug=True) # debug = True will print all intermediate data. 
    plot(ret, exaggeration=100, force_visual_scale = .5, zoom_factor = .8) # exaggeration is a multiplier to make the displacements visually distinct

if __name__ == "__main__":
    main()
