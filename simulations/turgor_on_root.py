
import fenics as fe
from bvpy.bvp import BVP
from granap.needle_class import NeedleAnatomy
from bvpy.utils.visu_pyvista import visualize
from bvpy.vforms import HyperElasticForm, LinearElasticForm
from bvpy.utils.pre_processing import HeterogeneousParameter
from bvpy.vforms.elasticity import StVenantKirchoffPotential
from bvpy.boundary_conditions.neumann import NormalNeumann
import pyvista as pv
from granap.root_class import RootAnatomy

class PreMarkedNormalNeumann(NormalNeumann):
    def markMeshFunction(self, mf):
        # Bypass overwriting the Gmsh pre-computed tags in bdata
        pass


params = []
params.append({"name": "planttype", "value": 1})
params.append({"name": "randomness", "value": 1.0, "smoothness": 0.3})
params.append({"name": "stele", "layer_diameter": 0.15,"cell_diameter": 0.01})
params.append({"name": "pericycle", "cell_diameter": 0.02, "cell_width": 0.015, "n_layers": 1, "order": 2, "shift": 5})
params.append({"name": "endodermis", "cell_diameter": 0.02, "cell_width": 0.05, "n_layers": 1, "order": 3, "shift": 5})
params.append({"name": "cortex", "cell_diameter": 0.045, "cell_width": 0.045, "n_layers": 3, "order": 4, "shift":10})
params.append({"name": "exodermis", "cell_diameter": 0.0225, "n_layers": 1, "order": 5})
params.append({"name": "epidermis", "cell_diameter": 0.02, "n_layers": 1, "order": 6})
params.append({"name": "xylem", "n_files": 1, "max_size": 0.065, "cell_diameter": 0.03, "ratio":6.5})
params.append({"name": "phloem", "n_files": 8, "cell_diameter": 0.01})

generic_organ = RootAnatomy(params)
generic_organ.generate_cells()

cwt = {
        "epidermis": 2,
        "exodermis": 2,
        "endodermis": 1.5,
        "cortex": 1,
        "vascular_parenchyma": 1,
        "pericycle": 1,
        "phloem": 1,
        "metaxylem": 2,
        "outerwall": 2,
    }

domain = generic_organ.to_domain(celldomain = False, 
                          cell_size=10, dim=2, symplast=False, clear=True,
                          simplify_tol = 0.02,
                          cell_wall_thickness = cwt)
print("Meshing...")
domain.discretize()
print("Meshing done!")
print("n cells:", domain.mesh.num_cells())
print("sub_domain_names:", domain.sub_domain_names)

# Define material properties
Young = 100
Poisson = .49

potential_energy = StVenantKirchoffPotential(young=Young, poisson=Poisson)
non_linear_response = HyperElasticForm(potential_energy=potential_energy, source=[0., 0.])


structural_test = BVP(domain=domain, vform=non_linear_response)
print("BVP created!")


zero_vec = fe.Constant((0.0, 0.0))
center_bc = fe.DirichletBC(
    structural_test.functionSpace,
    zero_vec,
    domain.bdata,
    domain.xylem_boundary_tag,
)
structural_test.dirichletCondition.append(center_bc)
print("Dirichlet BC added!")

pl = pv.Plotter()
visualize(structural_test, visu_type='dirichlet', val_range=[-0.5, 0.5], plotter=pl, show_plot=False)
pl.show()

turgor = PreMarkedNormalNeumann(val=-0.3, ind=domain.symplast_boundary_tag)
structural_test.add_boundary_condition(turgor)
print("Neumann BC added!")

print("Solving...")
structural_test.solve(
    linear_solver='mumps',
    report=True,
    krylov_solver={'absolute_tolerance': 1e-14},
    relative_tolerance=1e-7,
    absolute_tolerance=1e-6,
    preconditioner='none',
    maximum_iterations=500,
    line_search='bt'
)
print("Solving done!")

# ----- Utility Functions -----
def xdmf_save(path, solution, vform):
    solution.rename("Displacement Vector", "")
    strain = vform.get_strain(solution)
    strain.rename("Strain", "")
    stress = vform.get_stress(solution)
    stress.rename("Stress", "")
    xdmf_file = fe.XDMFFile(fe.MPI.comm_world, path)
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["functions_share_mesh"] = True
    xdmf_file.parameters["rewrite_function_mesh"] = False    
    xdmf_file.write(solution, 1)
    xdmf_file.write(strain, 1)
    xdmf_file.write(stress, 1)

xdmf_save("./turgor_on_root.xdmf", structural_test.solution, non_linear_response)
print("Done!")
