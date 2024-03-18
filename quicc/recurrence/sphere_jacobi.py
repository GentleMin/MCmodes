"""Module to compute the recurrence relations for the spectral Jacobi operators of the radial direction in a sphere"""

from __future__ import division
from __future__ import unicode_literals

import sympy

import quicc.recurrence.symbolic_jacobi as mod

l = sympy.Symbol('l')
n = sympy.Symbol('n')
w_alpha_legendre = 0
w_alpha_chebyshev = -sympy.Rational(1,2)

w_alpha = w_alpha_chebyshev
w_beta = l - sympy.Rational(1,2)

symbolic = mod.SymbolicJacobi(a = w_alpha, b = w_beta)

def chooseParameters(alpha, beta):
    """Choose the Jacobi polynomial to use"""

    global symbolic, w_alpha, w_beta
    w_alpha = alpha
    w_beta = beta
    symbolic = mod.SymbolicJacobi(a = w_alpha, b = w_beta)

def x1():
    """Sphere x operator"""

    # Setup terms in recurrence
    terms = [{'q':0, 'p':1, 'd':0, 'c':1}]
    r = symbolic.build_recurrence(terms, {0:1})

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")

def r2():
    """Sphere r^2 operator"""

    # Setup terms in recurrence
    terms = [{'q':0, 'p':1, 'd':0, 'c':sympy.Rational(1,2)},{'q':0, 'p':0, 'd':0, 'c':sympy.Rational(1,2)}]
    r = symbolic.build_recurrence(terms, {0:1})

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")

def r4():
    """Sphere r^4 operator"""

    # Setup terms in recurrence
    terms = [{'q':0, 'p':2, 'd':0, 'c':sympy.Rational(1,4)},{'q':0, 'p':1, 'd':0, 'c':sympy.Rational(1,2)},{'q':0, 'p':0, 'd':0, 'c':sympy.Rational(1,4)}]
    r = symbolic.build_recurrence(terms, {0:1})

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")

def i1():
    """Sphere i1 (i1r1) operator"""

    # Setup terms in recurrence
    terms = [{'q':1, 'p':0, 'd':0, 'c':1}]
    r = symbolic.build_recurrence(terms, {0:1})

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")

def i2():
    """Sphere i2 (i1r1i1r1) operator"""

    # Setup terms in recurrence
    terms = [{'q':2, 'p':0, 'd':0, 'c':1}]
    r = symbolic.build_recurrence(terms, {0:1})

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")

def i2divrdiff():
    """Sphere i2 ((i1r1)^2 1/r D) operator"""

    # Setup terms in recurrence
    terms = [{'q':2, 'p':0, 'd':1, 'c':4.0}]
    r = symbolic.build_recurrence(terms, {0:1})

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")

def i2lapl():
    """Sphere i2lapl (i1r1i1r1 lapl) operator"""

    # Setup terms in recurrence
    terms = [
            {'q':2, 'p':1, 'd':2, 'c':8},
            {'q':2, 'p':0, 'd':2, 'c':8},
            {'q':2, 'p':0, 'd':1, 'c':4*(2*l + 3)}
            ]
    r = symbolic.build_recurrence(terms, {0:1})

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")

def i4():
    """Sphere i4 (i1r1i1r1i1r1i1r1) operator"""

    # Setup terms in recurrence
    terms = [{'q':4, 'p':0, 'd':0, 'c':1}]
    r = symbolic.build_recurrence(terms, {0:1})

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")

def i4divrdiff():
    """Sphere i4 ((i1r1)^4 1/r D) operator"""

    # Setup terms in recurrence
    terms = [{'q':4, 'p':0, 'd':1, 'c':4.0}]
    r = symbolic.build_recurrence(terms, {0:1})

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")

def i4lapl():
    """Sphere i4lapl (i1r1i1r1i1r1i1r1 lapl) operator"""

    # Setup terms in recurrence
    terms = [
            {'q':4, 'p':1, 'd':2, 'c':8},
            {'q':4, 'p':0, 'd':2, 'c':8},
            {'q':4, 'p':0, 'd':1, 'c':4*(2*l + 3)}
            ]
    r = symbolic.build_recurrence(terms, {0:1})

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")

def i4lapl2():
    """Sphere i4lapl2 (i1r1i1r1i1r1i1r1 bilapl) operator"""

    # Setup terms in recurrence
    terms = [
            {'q':4, 'p':2, 'd':4, 'c':64},
            {'q':4, 'p':1, 'd':4, 'c':128},
            {'q':4, 'p':0, 'd':4, 'c':64},
            {'q':4, 'p':1, 'd':3, 'c':64*(2*l+5)},
            {'q':4, 'p':0, 'd':3, 'c':64*(2*l+5)},
            {'q':4, 'p':0, 'd':2, 'c':16*(2*l+3)*(2*l+5)}
            ]
    r = symbolic.build_recurrence(terms, {0:1})

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")

def i1qm():
    """Sphere i1qm (i1r1 coriolis Q(l-1)) operator"""

    # Compute starting terms
    r = symbolic.spectral_increase({0:-4}, True)

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")

def i1qp():
    """Sphere i2qp (i1r1 coriolis Q(l+1)) operator"""

    partA = symbolic.spectral_integral_decrease({0:(2*l+1)}, True)
    partB = symbolic.spectral_decrease({0:2}, True)

    r = partA
    for k,v in partB.items():
        r[k] = r[k] + v
        r[k] = r[k].simplify().factor()

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")

def i2qm():
    """Sphere i2qm (i1r1i1r1 coriolis Q(l-1)) operator"""

    # Compute starting terms
    fs = symbolic.spectral_increase({0:-4}, False)

    # Setup terms in recurrence
    terms = [
            {'q':1, 'p':0, 'd':0, 'c':1},
            ]
    r = symbolic.build_recurrence(terms, fs)

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")

def i2qp():
    """Sphere i2qp (i1r1i1r1 coriolis Q(l+1)) operator"""

    above = mod.SymbolicJacobi(a = w_alpha, b = w_beta + 1)

    # Setup terms in recurrence
    terms = [
            {'q':1, 'p':0, 'd':0, 'c':(2*l + 1)},
            ]
    tmp = above.build_recurrence(terms, {0:1}, False)
    partA = symbolic.spectral_decrease(tmp, True)

    # Compute starting terms
    fs = symbolic.spectral_decrease({0:-(2*l-1)}, False)

    # Setup terms in recurrence
    terms = [
            {'q':1, 'p':0, 'd':0, 'c':1},
            ]
    partB = symbolic.build_recurrence(terms, fs, True)
    r = partA
    for k,v in partB.items():
        r[k] = r[k] + v
        r[k] = r[k].simplify().factor()

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")

def i4qm():
    """Sphere i4qm (i1r1i1r1i1r1i1r1 coriolis Q(l-1)) operator"""

    # Compute starting terms
    fs = symbolic.spectral_increase({0:-4}, False)

    # Setup terms in recurrence
    terms = [
            {'q':3, 'p':0, 'd':0, 'c':1},
            ]
    r = symbolic.build_recurrence(terms, fs)

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")

def i4qp():
    """Sphere i4qp (i1r1i1r1i1r1i1r1 coriolis Q(l+1)) operator"""

    above = mod.SymbolicJacobi(a = w_alpha, b = w_beta + 1)

    # Setup terms in recurrence
    terms = [
            {'q':1, 'p':0, 'd':0, 'c':(2*l + 1)},
            ]
    tmp = above.build_recurrence(terms, {0:1}, False)
    tmp2 = symbolic.spectral_decrease(tmp, False)
    terms = [
            {'q':2, 'p':0, 'd':0, 'c':1},
            ]
    partA = symbolic.build_recurrence(terms, tmp2, True)

    # Compute starting terms
    fs = symbolic.spectral_decrease({0:-(2*l-1)}, False)

    # Setup terms in recurrence
    terms = [
            {'q':3, 'p':0, 'd':0, 'c':1},
            ]
    partB = symbolic.build_recurrence(terms, fs, True)
    r = partA
    for k,v in partB.items():
        r[k] = r[k] + v
        r[k] = r[k].simplify().factor()

    # Print recurrence relation per diagonals
    for k,rec in sorted(r.items()):
        print("\t" + str(k) + ": \t" + str(rec))
    print("\n")
