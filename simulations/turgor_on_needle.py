
import fenics as fe
from bvpy.bvp import BVP
from granap.needle_class import NeedleAnatomy
from bvpy.utils.visu_pyvista import visualize
from bvpy.vforms import HyperElasticForm
from bvpy.vforms.elasticity import StVenantKirchoffPotential
from bvpy.boundary_conditions.neumann import NormalNeumann
import pyvista as pv

class PreMarkedNormalNeumann(NormalNeumann):
    def markMeshFunction(self, mf):
        # Bypass overwriting the Gmsh pre-computed tags in bdata
        pass

# Physical Group tags defined in OrganDomain are now accessed via domain attributes
needle = NeedleAnatomy()
needle.update_params("resin_duct", "n_files", 0)
needle.update_params("stomata", "n_files", 2)
needle.generate_cells()
needle.plot_cells()

cwt = {
        "epidermis": 2,
        "hypodermis": 2,
        "endodermis": 1.5,
        "mesophyll": 1,
        "parenchyma": 1,
        "phloem": 1,
        "duct": 5,
        "outerwall": 2,
        "cambium": 1,
        "guard cell": 2,
        "Strasburger cell": 1,
        "xylem": 1.5,
        "air space": 0.001,
        "pore": 0.001
    }
domain = needle.to_domain(celldomain = False, 
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

pl = pv.Plotter()
visualize(structural_test, visu_type='domain',plotter=pl, show_plot=False)
pl.camera_position = 'xy'
pl.show()

# ---- Boundary conditions using pre-marked Physical Groups from OrganDomain ----
#
# domain.bdata is a MeshFunction already tagged by Gmsh Physical Groups:
#   tag 3 (_CENTER_BOUNDARY_TAG) → "center" (innermost cell boundary)
#   tag 1 (_CELLS_BOUNDARY_TAG)  → "cells"  (all cell walls)
#   tag 2 (_AIR_BOUNDARY_TAG)    → "air_space"

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
pl.camera_position = 'xy'
pl.show()

# -- Normal Neumann (turgor pressure) on the all cell walls --
# NormalNeumann(val, ind=tag) → bvp integrates val * n * ds(tag)
print("Adding Neumann BC to symplast boundary...")
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

print("Saving to XDMF...")
xdmf_save("./turgor_on_needle.xdmf", structural_test.solution, non_linear_response)
print("Done!")