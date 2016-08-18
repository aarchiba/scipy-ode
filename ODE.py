from __future__ import division

import copy

import numpy as np
import scipy.integrate
import scipy.interpolate

class RHS(object):
    def __init__(self):
        pass

    def __call__(self, t, y):
        raise NotImplementedError

class CallableRHS(RHS):
    def __init__(self, callable):
        self.callable = callable

    def __call__(self, t, y):
        return self.callable(t,y)

class InvalidEvaluationPoint(Exception):
    """The user function cannot be evaluated here

    If user right-hand side functions have domain restrictions, they can
    indicate that a trial point is outside that domain by raising this
    exception. Integrators can then try to continue, reducing the step size
    and hoping to avoid the domain limits; if the solution cannot be
    continued without triggering this exception the integrator will report
    failure.
    """

class IntegrationFailure(ValueError):
    """Integrator was unable to integrate to the requested point"""

class Solver(object):
    def __init__(self, t0, y0, rhs, t_increasing=True, atol=1e-8, rtol=1e-8):
        if isinstance(rhs, RHS):
            self._rhs = rhs
        elif callable(rhs):
            self._rhs = CallableRHS(rhs)
        else:
            raise ValueError("RHS object not understood: %s" % trhs)
        self.t_increasing = bool(t_increasing)
        self.dir_factor = 1 if self.t_increasing else -1
        self.t0 = float(t0)
        self.y0 = np.asarray(y0, dtype=float) # FIXME: complex
        self.t = self.t0
        self.y = self.y0
        self.nfev = 0
        self.f = self.rhs(self.t, self.y)
        self.dim = len(self.y0)
        self.t_last = self.t
        self.y_last = self.y
        self.f_last = self.f
        self.atol = atol # FIXME: ensure float or array like y0
        self.rtol = rtol

    def rhs(self, t, y):
        self.nfev += 1
        return self._rhs(t,y)

    def _err(self, y1, y2):
        """Return a scaled estimate of how close y1 and y2 are

        Compares them in relative and absolute tolerance; iff the result is
        smaller than one they're pretty close.
        """
        scale = self.atol + np.hypot(y1,y2)*self.rtol
        return np.sqrt(np.mean(((y2-y1)/scale)**2))

    def advance(self, max_t=None):
        raise NotImplementedError

    def _evaluate(self, t, derivative, state):
        raise NotImplementedError

    def value(self, t=None, derivative=0, state=None):
        if state is None:
            state = self.state
        if t is None:
            t = self.t
        if self.dir_factor*t<self.dir_factor*state.t_last:
            raise ValueError("Evaluation time %g before "
                                 "beginning of current step (%g)"
                                 % (t, state.t_last))
        if self.dir_factor*t>self.dir_factor*state.t:
            raise ValueError("Evaluation time %g after "
                                 "end of current step (%g)"
                                 % (t, state.t),
                                 t,state.t_last,state.t,self.dir_factor)
        return self._evaluate(t, derivative, state)

    @property
    def state(self):
        raise NotImplementedError

def _make_tableau(dim, n):
    return [np.nan+np.empty((j+1,dim)) for j in range(n)]
def _fill_tableau_row(tableau, new_values, row, sizes):
    j = row
    Y = tableau
    Y[j][0] = new_values
    for k in range(1,j+1):
        Y[j][k] = (Y[j][k-1]
            + (Y[j][k-1]-Y[j-1][k-1])
                /(sizes[j-k]/sizes[j]-1))

class BSSolver(Solver):
    def __init__(self, t0, y0, rhs, t_increasing=True,
                     first_step=1e-4, max_order=9):
        Solver.__init__(self, t0, y0, rhs, t_increasing=t_increasing)
        self.h = self.dir_factor*float(first_step)
        self.h_next = self.h
        self.ns = np.array([2+4*i for i in range(max_order)])
        self.scales = self.ns.astype(float)**(-2)
        self.Y = _make_tableau(self.dim, len(self.ns))
        self.ds = [_make_tableau(self.dim, len(self.ns))]
        for j in range(len(self.ns)):
            self.ds.append(_make_tableau(self.dim, len(self.ns)-j))
            self.ds.append(_make_tableau(self.dim, len(self.ns)-j))
        self.order_target = len(self.ns)-2
        self.atol = 1e-8
        self.rtol = 1e-8

    class State(object):
        pass

    def _modified_midpoint(self, n, j):
        h = self.h/n
        ys = np.zeros((n+2, len(self.y)))
        fs = np.zeros((n+2, len(self.y)))
        ys[0] = self.y
        fs[0] = self.rhs(self.t, ys[0])
        ys[1] = ys[0] + h*fs[0]
        for i in range(1,n+1):
            fs[i] = self.rhs(self.t+i*h, ys[i])
            ys[i+1] = ys[i-1] + 2*h*fs[i]

        fs[n+1] = self.rhs(self.t, ys[n+1])
        ds = np.zeros((2*j+1, len(self.y)))
        ds[0] = ys[n//2]
        ds[1] = fs[n//2]
        if n>2:
            ifs = fs[n//2-2*j+1:n//2+2*j]
        else:
            ifs = fs.copy()
        for i in range(2, 2*j+1):
            # take central differences
            ifs = (ifs[2:] - ifs[:-2])/(2*h)
            ds[i] = ifs[ifs.shape[0]//2]

        return (1/4)*(ys[n+1] + 2*ys[n] + ys[n-1]), ds

    def advance(self):
        """Advance one step of appropriate step size"""
        self._advance()

    def _advance(self):
        """Try to advance one fixed-size step and adjust step size

        Return whether the step succeeded; whether or not it did,
        modify stepper parameters so that the next call is sensible
        """
        # FIXME: determine sensible order and error estimate
        for j in range(len(self.ns)):
            Y, ds = self._modified_midpoint(self.ns[j], j+1)
            _fill_tableau_row(self.Y, Y, j, self.scales)
            _fill_tableau_row(self.ds[0], ds[0], j, self.scales)
            for k in range(j+1):
                _fill_tableau_row(self.ds[2*k+1], ds[2*k+1], j-k,
                                      self.ns[k:].astype(float)**(-2))
                _fill_tableau_row(self.ds[2*k+2], ds[2*k+2], j-k,
                                      self.ns[k:].astype(float)**(-2))
            err = self._err(self.Y[j][j], self.Y[j][j-1])

        order = j-1

        t_next = self.t+self.h
        y_next = self.Y[order][order]
        f_next = self.rhs(t_next, y_next)

        S = BSSolver.State()
        S.t_last = self.t
        S.t = t_next

        # Note you can use lower-degree polynomials here, and might want to
        kx = [0,0] + [0.5]*(2*order+1) + [1,1]
        poly_degree = len(kx)-1
        mu = poly_degree - 4
        ky = [self.y, self.h*self.f]
        ky.append(self.ds[0][order][order])
        for k in range(order):
            ky.append(self.h**(2*k+1)*self.ds[2*k+1][order-k][order-k])
            ky.append(self.h**(2*k+2)*self.ds[2*k+2][order-k][order-k])
        ky += [y_next, self.h*f_next]
        kx = np.array(kx)
        ky = np.array(ky)
        print("Making polynomial:")
        print(kx)
        print(ky)
        S.poly = scipy.interpolate.KroghInterpolator(kx,ky)
        self._state = S
        self.t_last, self.t = self.t, t_next
        self.y_last, self.y = self.y.copy(), y_next.copy()
        self.f_last, self.f = self.f, f_next
        print("Advanced from %g to %g" % (self.t_last, self.t))
        print("Y from %s to %s" % (self.y_last, self.y))
        print(self.Y[j][j], self.Y[j][j-1])
        return err

    def _evaluate(self, t, derivative, state):
        tp = (t-state.t_last)/(state.t-state.t_last)
        return ((state.t-state.t_last)**(-derivative)
                    *state.poly.derivative(tp, derivative))

    @property
    def state(self):
        return copy.deepcopy(self._state)




class Solution(object):

    def __init__(self, t0, y0, rhs, solver_type=None, **kwargs):
        self.t0 = float(t0)
        self.y0 = np.asarray(y0, dtype=float)
        self.solver_increasing = solver_type(self.t0,
                                            self.y0,
                                            rhs,
                                            t_increasing=True,
                                            **kwargs)
        self.solver_decreasing = solver_type(self.t0,
                                            self.y0,
                                            rhs,
                                            t_increasing=False,
                                            **kwargs)
        self.ts_increasing = [self.t0]
        self.states_increasing = []
        self.ts_decreasing = [self.t0]
        self.states_decreasing = []

    def extend(self, t):
        t = float(t)
        if t>=self.t0:
            while self.solver_increasing.t < t:
                self.solver_increasing.advance()
                self.states_increasing.append(self.solver_increasing.state)
                self.ts_increasing.append(self.solver_increasing.t)
        elif t<self.t0:
            while self.solver_decreasing.t > t:
                self.solver_decreasing.advance()
                self.states_decreasing.append(self.solver_decreasing.state)
                self.ts_decreasing.append(self.solver_decreasing.t)
        else:
            raise ValueError("Attempting to evaluate solver out to %s" % t)

    def __call__(self, t, derivative=0):
        self.extend(t)
        if t>=self.t0:
            i = np.searchsorted(self.ts_increasing,t)-1
            return self.solver_increasing.value(
                t,
                state=self.states_increasing[i],
                derivative=derivative)
        elif t<self.t0:
            i = np.searchsorted(self.ts_decreasing[::-1],t)-1
            #print(t,self.ts_decreasing[::-1],i)
            return self.solver_decreasing.value(
                t,
                state=self.states_decreasing[::-1][i],
                derivative=derivative)
        else:
            raise AssertionError("Extend error checking failed")

def solve(t0, y0, rhs, ts, SolverType=None):
    return Solver(t0, y0, rhs, SolverType=SolverType)(ts)
