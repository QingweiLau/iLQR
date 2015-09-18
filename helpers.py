import casadi as ca

__author__ = 'belousov'


def policy_cost(x, u, dt, l, lf):
    n_sim = len(u[:])
    [final_cost] = lf([x[n_sim]])
    J = final_cost
    for k in range(n_sim):
        [stage_cost] = l([x[k], u[k], dt])
        J += stage_cost
    return J


def fwd(x_all, u_all, k_all, K_all, alpha, F, dt, state, control):
    n_sim = len(u_all[:])
    xk = x_all[0]
    x_new = [xk]
    u_new = []
    for k in range(n_sim):
        uk = u_all[k] + alpha * k_all[k] + ca.mul(K_all[k], xk - x_all[k])
        u_new.append(uk)
        [xk_next] = F([xk, uk, dt])
        x_new.append(xk_next)
        xk = xk_next
    x_new = state.repeated(ca.horzcat(x_new))
    u_new = control.repeated(ca.horzcat(u_new))
    return [x_new, u_new]