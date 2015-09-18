from __future__ import division

# Numerics
import numpy as np

# CasADi
import casadi as ca
import casadi.tools as cat

# Module functions
from plotting import plot_policy

# Configuration
np.set_printoptions(suppress=True, precision=4)

__author__ = 'belousov'


# ----------------------------------------------------------------------------
#                               Parameters
# ----------------------------------------------------------------------------

# Simulation
t_sim = 2.0  # simulation time
n_sim = 20  # time steps
dt = t_sim / n_sim  # time-step length

# Model
nx = 3
nu = 2

# Initial condition
x0 = ca.DMatrix([10, 6, ca.pi])

# Identity matrix
Inx = ca.DMatrix.eye(nx)


# ----------------------------------------------------------------------------
#                                 Dynamics
# ----------------------------------------------------------------------------

# Continuous dynamics
dt_sym = ca.SX.sym('dt')
state = cat.struct_symSX(['x', 'y', 'phi'])
control = cat.struct_symSX(['v', 'w'])
rhs = cat.struct_SX(state)
rhs['x'] = control['v'] * ca.cos(state['phi'])
rhs['y'] = control['v'] * ca.sin(state['phi'])
rhs['phi'] = control['w']
f = ca.SXFunction('Continuous dynamics', [state, control], [rhs])

# Discrete dynamics
state_next = state + dt_sym * f([state, control])[0]
op = {'input_scheme': ['state', 'control', 'dt'],
      'output_scheme': ['state_next']}
F = ca.SXFunction('Discrete dynamics',
                  [state, control, dt_sym], [state_next], op)
Fj_x = F.jacobian('state')
Fj_u = F.jacobian('control')
F_xx = ca.SXFunction('F_xx', [state, control, dt_sym],
                     [ca.jacobian(F.jac('state')[i, :].T, state) for i in
                      range(nx)])
F_uu = ca.SXFunction('F_uu', [state, control, dt_sym],
                     [ca.jacobian(F.jac('control')[i, :].T, control) for i in
                      range(nx)])
F_ux = ca.SXFunction('F_ux', [state, control, dt_sym],
                     [ca.jacobian(F.jac('control')[i, :].T, state) for i in
                      range(nx)])

# Cost functions
Qf = ca.diagcat([1., 1., 0.])
final_cost = 0.5 * ca.mul([state.cat.T, Qf, state.cat])
op = {'input_scheme': ['state'],
      'output_scheme': ['cost']}
lf = ca.SXFunction('Final cost', [state], [final_cost], op)

R = ca.DMatrix.eye(nu) * dt * 1e-2
cost = 0.5 * ca.mul([control.cat.T, R, control.cat])
op = {'input_scheme': ['state', 'control', 'dt'],
      'output_scheme': ['cost']}
l = ca.SXFunction('Running cost', [state, control, dt_sym], [cost], op)

# Quadratic expansions of costs
lfg = lf.gradient('state')          # lf_x
lfh = lf.hessian('state')           # lf_xx

lg_x = l.gradient('state')          # l_x
lh_x = l.hessian('state')           # l_xx
lg_u = l.gradient('control')        # l_u
lh_u = l.hessian('control')         # l_uu
lgj_ux = ca.SXFunction('l_ux_fun',  # l_ux
                       [state, control, dt_sym],
                       [ca.jacobian(l.grad('control'), state)])


# ----------------------------------------------------------------------------
#                               Optimization
# ----------------------------------------------------------------------------

# Degrees of freedom for the optimizer
V = cat.struct_symSX([
    (
        cat.entry('X', repeat=n_sim + 1, struct=state),
        cat.entry('U', repeat=n_sim, struct=control)
    )
])

# Multiple shooting constraints
g = []
for k in range(n_sim):
    g.append(V['X', k + 1] - F([V['X', k], V['U', k], dt])[0])
g = ca.vertcat(g)

# Objective
[final_cost] = lf([V['X', n_sim]])
J = final_cost

# Regularize controls
for k in range(n_sim):
    [stage_cost] = l([V['X', k], V['U', k], dt])
    J += stage_cost

# Formulate the NLP
nlp = ca.SXFunction('nlp', ca.nlpIn(x=V), ca.nlpOut(f=J, g=g))

# Create solver
opts = {'linear_solver': 'ma57'}
solver = ca.NlpSolver('solver', 'ipopt', nlp, opts)

# Constraints
lbx = V(-ca.inf)
ubx = V(ca.inf)

# x(t=0) = x0
lbx['X', 0] = ubx['X', 0] = x0

# Solve nlp
sol = solver(x0=0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)
sol = V(sol['x'])

# Plot policy
plot_policy(sol['X', :, 'x'], sol['X', :, 'y'], sol['X', :, 'phi'],
            sol['U', :, 'v'], sol['U', :, 'w'], np.linspace(0, t_sim, n_sim+1))
