from __future__ import division

# Plotting
import matplotlib.pyplot as plt

# Numerics
import numpy as np

# CasADi
import casadi as ca
import casadi.tools as cat

# Module functions
from plotting import plot_policy
from helpers import policy_cost, fwd

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
t = np.linspace(0, t_sim, n_sim+1)

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
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plot_policy(ax, sol['X', :, 'x'], sol['X', :, 'y'], sol['X', :, 'phi'],
            sol['U', :, 'v'], sol['U', :, 'w'], t)


# ----------------------------------------------------------------------------
#                                   iLQR
# ----------------------------------------------------------------------------

# Play nominal policy
u_all = control.repeated(ca.DMatrix.zeros(nu,n_sim))
u_all[:,'v'] = 5
u_all[:,'w'] = 1
xk = x0
x_all = [xk]
for k in range(n_sim):
    [xk_next] = F([ xk, u_all[k], dt ])
    x_all.append(xk_next)
    xk = xk_next
x_all = state.repeated(ca.horzcat(x_all))

J0 = policy_cost(x_all,u_all,dt,l,lf)

# Plot policy
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plot_policy(ax, x_all[:,'x'], x_all[:,'y'], x_all[:,'phi'],
            u_all[:,'v'], u_all[:,'w'], t)

# Iterations
mu = 0; mu_min = 1e-6; d_mu0 = 2; d_mu = d_mu0;
mu_flag = False

while True:

    # Backward pass with regularization
    while True:
        k_all = [None] * n_sim
        K_all = [None] * n_sim

        # Compute final V-function expansion
        [Vx, _] = lfg([ x_all[n_sim] ])
        [Vxx, _, _] = lfh([ x_all[n_sim] ])

        for i in range(n_sim-1, -1, -1):
            # print '---- ' + str(i) + ' ----'
            # Common argument
            argument = [ x_all[i], u_all[i], dt ]

            # Expand F
            [Fx, _] = Fj_x(argument)
            [Fu, _] = Fj_u(argument)
            Fxx = F_xx(argument)
            Fuu = F_uu(argument)
            Fux = F_ux(argument)

            # Expand running cost
            [lx, _] = lg_x(argument)
            [lu, _] = lg_u(argument)
            [lxx, _, _] = lh_x(argument)
            [luu, _, _] = lh_u(argument)
            [lux] = lgj_ux(argument)

            # Expand Q-function
            Qx = lx + ca.mul(Fx.T, Vx)
            Qu = lu + ca.mul(Fu.T, Vx)
            Qxx = lxx + ca.mul([ Fx.T, Vxx + mu*Inx, Fx ])
            # + sum([a*b for (a,b) in zip(Fxx,Vx)])
            Quu = luu + ca.mul([ Fu.T, Vxx + mu*Inx, Fu ])
            # + sum([a*b for (a,b) in zip(Fuu,Vx)])
            Qux = lux + ca.mul([ Fu.T, Vxx + mu*Inx, Fx ])
            # + sum([a*b for (a,b) in zip(Fux,Vx)])

            # print 'Quu:'
            # print Quu
            # print 'Fu:'
            # print Fu
            # print 'Vxx:'
            # print Vxx

            # Check if Quu is positive definite
            try:
                np.linalg.cholesky(Quu)
            except np.linalg.LinAlgError as e:
                mu_flag = True
                print 'Quu:'
                print Quu
                break

            # Save gains
            iQuu = ca.inv(Quu)
            k_all[i] = -ca.mul(iQuu, Qu)
            K_all[i] = -ca.mul(iQuu, Qux)

            # Compute expansion of V-function
            # Vx = Qx - ca.mul([ Qux.T, iQuu, Qu ])
            # Vxx = Qxx - ca.mul([ Qux.T, iQuu, Qux ])
            Vx = Qx + ca.mul([ K_all[i].T, Quu, k_all[i] ]) + \
                ca.mul(K_all[i].T, Qu) + ca.mul(Qux.T, k_all[i])
            Vxx = Qxx + ca.mul([ K_all[i].T, Quu, K_all[i] ]) + \
                ca.mul(K_all[i].T, Qux) + ca.mul(Qux.T, K_all[i])

        # Manage regularization
        if mu_flag:
            # Increase mu
            d_mu = max(d_mu0, d_mu*d_mu0)
            mu = max(mu_min, mu*d_mu)
            mu_flag = False
            print 'mu had to be increased to: ' + str(mu)
        else:
            # Decrease mu
            d_mu = min(1/d_mu0, d_mu/d_mu0)
            if mu*d_mu > mu_min:
                mu = mu * d_mu
            else:
                mu = 0
            break

    # Forward pass with line search
    alpha = 1.0
    J1 = J0
    while True:
        [x_new, u_new] = fwd(x_all, u_all, k_all, K_all,
                             alpha, F, dt, state, control)
        J = policy_cost(x_new, u_new, dt, l, lf)
        # print 'J0 = ' + str(J0)
        # print 'J1 = ' + str(J1)
        # print 'J  = ' + str(J)
        if J < J1:
            if (J1-J)/J1 > 0.01:
                alpha = alpha/2
                J1 = J
            else:
                x_all, u_all = x_new, u_new
                J1 = J
                break
        else:
            if J1 == J0:
                alpha = alpha/2
            else:
                [x_all, u_all] = fwd(x_all, u_all, k_all, K_all,
                                     alpha*2, F, dt, state, control)
                J = J1
                break
        # if alpha < 1e-10:
        #     x_all, u_all = x_new, u_new
        #     J0 = J1 = J
        #     break
        # alpha = alpha/2

    # Plot policy
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plot_policy(ax, x_all[:,'x'], x_all[:,'y'], x_all[:,'phi'],
                u_all[:,'v'], u_all[:,'w'], t)

    # Current best cost
    print 'J0 = ' + str(J)

    # Finish if improvement is tiny
    if (J0-J)/J0 < 0.01:
        J0 = J
        break
    else:
        J0 = J





























