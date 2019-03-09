from sympy import pprint
from sympy import Symbol
from sympy import Eq
from sympy import simplify
from sympy.solvers import solve

ex_new = Symbol("E_x|t+dt; i,j,k")
ex_old = Symbol("E_x|t; i,j,k")
dt = Symbol("dt")
dx = Symbol("dx")
dy = Symbol("dy")
dz = Symbol("dz")
sigma_x = Symbol("o`x")
sigma_y = Symbol("o`y")
sigma_z = Symbol("o`z")
e0 = Symbol("e0")
c0 = Symbol("c0")
integral_e = Symbol("E_Sum")
integral_curl = Symbol("Curl_Sum")
exx = Symbol("exx")
hz = Symbol("H_z|t+dt/2; i,j,k")
hy = Symbol("H_y|t+dt/2; i,j,k")
hz_dy = Symbol("H_z|t+dt/2; i,j-1,k")
hy_dz = Symbol("H_y|t+dt/2; i,j,k-1")

equation = Eq((ex_new - ex_old) / dt + ((sigma_z + sigma_y) / e0) * ex_old + ((sigma_z * sigma_y * dt) / e0**2) * integral_e,
        (c0 / exx) * (((hz - hz_dy) / dy) - ((hy - hy_dz) / dz)) + ((c0 * sigma_x * dt) / (exx * e0)) * integral_curl)
pprint(equation)
solved = solve(equation, ex_new)
pprint(solved)
