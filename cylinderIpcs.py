import fenics as fe
from dolfin import *
from classnsipcs import IPCSSolver
from tqdm import tqdm
from mpi4py import MPI
#######################################################################################################################
# Get the global communicator
comm = MPI.COMM_WORLD
# Get the rank of the process
rank = comm.Get_rank()
# Get the size of the communicator (total number of processes)
size = comm.Get_size()
fe.set_log_level(fe.LogLevel.ERROR)
#######################################################################################################################
# Define boundary classes for inlet, outlet, walls, and cylinder
class InletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0) and on_boundary  # Inlet at x = 0
class OutletBoundary(SubDomain):
    def __init__(self, L, **kwargs):
        super().__init__(**kwargs)
        self.L = L  # Length of the domain

    def inside(self, x, on_boundary):
        return near(x[0], self.L) and on_boundary  # Outlet at x = L
class WallsBoundary(SubDomain):
    def __init__(self, H, **kwargs):
        super().__init__(**kwargs)
        self.H = H  # Height of the domain

    def inside(self, x, on_boundary):
        # Bottom wall and top wall
        return (near(x[1], 0) or near(x[1], self.H)) and on_boundary
# class CylinderBoundary(SubDomain):
#     def __init__(self, x_cyl, y_cyl, r, **kwargs):
#         super().__init__(**kwargs)
#         self.x_cyl = x_cyl  # Cylinder center x-coordinate
#         self.y_cyl = y_cyl  # Cylinder center y-coordinate
#         self.r = r  # Cylinder radius

#     def inside(self, x, on_boundary):
#         # Boundary of the cylinder, defined as a circle
#         return  near((x[0] - self.x_cyl)**2 + (x[1] - self.y_cyl)**2, self.r**2)
# Load the mesh
mesh = fe.Mesh("cylinder_flow.xml")
hmin = mesh.hmin()  # Minimum element size in the mesh
hmax = mesh.hmax()  # Maximum element size in the mesh
if rank == 0:
    print("Minimum dx:", hmin)
    print("Maximum dx:", hmax)
# Geometry parameters
L = 2.2  # Length of the domain (inlet to outlet)
H = 0.41  # Height of the domain (walls)
x_cyl, y_cyl, r = 0.2, 0.2, 0.05  # Cylinder center and radius
# Define the boundary markers (this should match your Gmsh boundary definitions)
boundaries = fe.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)  # MeshFunction for marking boundaries
# Mark the boundaries
inlet = InletBoundary()
outlet = OutletBoundary(L)
walls = WallsBoundary(H)
# cylinder = CylinderBoundary(x_cyl, y_cyl, r)
# Mark the boundaries with different integer numbers
inlet.mark(boundaries, 1)
outlet.mark(boundaries, 2)
walls.mark(boundaries, 3)
# cylinder.mark(boundaries, 3) # wall
inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')
expression_inflow = Expression(inflow_profile, degree=2)
# Define the inflow, outflow, and walls markers in the parameter dictionary
parameters_dict = {
    "inflow_number": 1,
    "walls_number": 3,  # Correct this to match walls
    "outflow_number": 2,  # Correct this to match outlet
    "boundaries": boundaries,
    "inflow_velocity": expression_inflow,  # Inflow velocity
    "mu": 1e-3,   # Viscosity
    "dt": 0.001,   # Time step
    "nut_": fe.Constant(1E-10),  # Turbulent viscosity
    "rho": 1.0,  # Density
}
# Create the Navier-Stokes solver
ns_problem = IPCSSolver(mesh, parameters_dict)
# Create function spaces and set boundary conditions
ns_problem.func_space()
# Define the initial conditions
ns_problem.inco()
# Define the boundary conditions
ns_problem.BC()
# Define the Navier-Stokes form
ns_problem.form()
#######################################################################################################################
# Name the variables for ParaView output and write to XDMF file
u_n, p_n = ns_problem.u_n, ns_problem.p_
u_n.rename("Velocity", "u")  
p_n.rename("Pressure", "p")
T = 50.0  # Final time
t = 0.0
num_steps = T / parameters_dict["dt"]
dt = parameters_dict["dt"]
it = 0
xdmf_file = fe.XDMFFile( "navier_stokes_cylinder_IPCS.xdmf")
xdmf_file.parameters["rewrite_function_mesh"] = True
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True
for i in tqdm(range(int(num_steps))):

    if it % 100 == 0:
        u_n, p_n = ns_problem.u_n, ns_problem.p_
        u_n.rename("Velocity", "u")  
        p_n.rename("Pressure", "p")
        xdmf_file.write(u_n, t)  # Write velocity to XDMF
        xdmf_file.write(p_n, t)  # Write pressure to XDMF
        xdmf_file.close()

    if it % 100 == 0 and rank == 0:
        print(f"Time step {it}", flush=True)

    it +=1
    t += dt
    # Solve the Navier-Stokes problem and update previous steps
    ns_problem.solve()

