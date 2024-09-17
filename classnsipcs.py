import fenics as fe

class IPCSSolver:
    def __init__(self, mesh, parameters_dict):
        # Initialize mesh and parameters
        self.mesh = mesh
        self.parameters_dict = parameters_dict
        self.dt = fe.Constant(parameters_dict["dt"]) # Time step
        self.mu = fe.Constant(parameters_dict["mu"]) # Viscosity
        self.rho = fe.Constant(parameters_dict["rho"]) # Density
        
    def create_function_spaces(self, degree=2):
        # Define the function spaces for velocity (vector) and pressure (scalar)
        V = fe.VectorFunctionSpace(self.mesh, 'P', degree)
        Q = fe.FunctionSpace(self.mesh, 'P', degree - 1)
        
        # Assign to self for later use
        self.V = V
        self.Q = Q
        
        # Define trial and test functions
        self.u = fe.TrialFunction(V)
        self.v = fe.TestFunction(V)
        self.p = fe.TrialFunction(Q)
        self.q = fe.TestFunction(Q)
        
        # Define previous and current time step functions
        self.u_n = fe.Function(V) # Previous time step
        self.u_ = fe.Function(V)  # Current time step`
        self.p_n = fe.Function(Q) # Previous time step
        self.p_ = fe.Function(Q)   # Current time step
        
    def define_boundary_condition(self):

        # Define boundary conditions for velocity and pressure
        inflow_velocity = self.parameters_dict["inflow_velocity"]
        inflow_number = self.parameters_dict["inflow_number"]
        walls_number = self.parameters_dict["walls_number"]
        outflow_number = self.parameters_dict["outflow_number"]
        boundaries = self.parameters_dict["boundaries"]
        # Define the boundary conditions
        bc_inflow = fe.DirichletBC(self.V, inflow_velocity, boundaries, inflow_number)
        bc_walls = fe.DirichletBC(self.V, fe.Constant((0, 0)), boundaries, walls_number)
        bc_outflow = fe.DirichletBC(self.Q, fe.Constant(0), boundaries, outflow_number)
        # Save boundary conditions for velocity and pressure for use in the solver steps
        self.bcu = [bc_inflow, bc_walls]
        self.bcp = bc_outflow

    def epsilon(self, u):  

        return fe.sym(fe.nabla_grad(u))

    def sigma(self, u, p):

        return 2*self.mu*self.epsilon(u) - p*fe.Identity(len(u))

    def define_form(self):
        # Define constants
        dt = fe.Constant(self.dt)
        rho = fe.Constant(self.rho)
        mu = self.mu
        n = fe.FacetNormal(self.mesh)
        # Tentative velocity step (solve for u*)
        U = 0.5 * (self.u_ + self.u)  # Average velocity
        f = fe.Constant((0, 0))  # Assuming no external force
        # Define variational problem for step 1
        F1 = (
                rho*fe.dot((self.u - self.u_n) / dt, self.v)*fe.dx  
            + rho*fe.dot(fe.dot(self.u_n, fe.nabla_grad(self.u_n)), self.v)*fe.dx  
            + fe.inner(self.sigma(U, self.p_n), self.epsilon(self.v))*fe.dx  
            + fe.dot(self.p_n*n, self.v)*fe.ds 
            - fe.dot(self.mu*fe.nabla_grad(U)*n, self.v)*fe.ds  
            - fe.dot(f, self.v)*fe.dx
        )
        self.a1 = fe.lhs(F1)
        self.L1 = fe.rhs(F1)  

        # Define variational problem for step 2
        self.a2 = fe.dot(fe.nabla_grad(self.p), fe.nabla_grad(self.q))*fe.dx
        self.L2 = fe.dot(fe.nabla_grad(self.p_n), fe.nabla_grad(self.q))*fe.dx - (1/dt)*fe.div(self.u_)*self.q*fe.dx
  
        # Define variational problem for step 3
        self.a3 = fe.dot(self.u, self.v)*fe.dx
        self.L3 = fe.dot(self.u_, self.v)*fe.dx - dt*fe.dot(fe.nabla_grad(self.p_ - self.p_n), self.v)*fe.dx

    def solve(self):

        A1 = fe.assemble(self.a1)
        b1 = fe.assemble(self.L1)
        [bc.apply(A1) for bc in self.bcu]
        [bc.apply(b1) for bc in self.bcu]
        fe.solve(A1, self.u_.vector(), b1, 'bicgstab', 'hypre_amg')
        # Step 2: Pressure correction step
        A2 = fe.assemble(self.a2)
        b2 = fe.assemble(self.L2)
        self.bcp.apply(A2)
        self.bcp.apply(b2)
        fe.solve(A2, self.p_.vector(), b2, 'bicgstab', 'hypre_amg')
        # Step 3: Velocity correction step
        A3 = fe.assemble(self.a3)
        b3 = fe.assemble(self.L3)
        fe.solve(A3, self.u_.vector(), b3, "cg", "sor")
        # Update previous time steps
        self.u_n.assign(self.u_)
        self.p_n.assign(self.p_)

    def define_initial_conditions(self):

        class InitialConditions_ns(fe.UserExpression):

            def __init__(self, **kwargs):
                super().__init__(**kwargs)  

            def eval(self, values, x):
                values[0] = 0.0  # Initial x-component of velocity
                values[1] = 0.0  # Initial y-component of velocity

            def value_shape(self):
                return (2,)
        class InitialConditions_p(fe.UserExpression):

            def __init__(self, **kwargs):
                super().__init__(**kwargs)  

            def eval(self, values, x):
                values[0] = 0.0

            def value_shape(self): 
                return ()
        
                # Define the initial conditions for velocity and pressure
        
        self.initial_conditions_ns = InitialConditions_ns(degree=2)
        self.u_.interpolate(self.initial_conditions_ns)
        self.initial_conditions_p = InitialConditions_p(degree=2)
        self.p_.interpolate(self.initial_conditions_p)


