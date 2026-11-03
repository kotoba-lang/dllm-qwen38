"""Analytic boundaries, published calibration and finite-grid coverage checks."""
import math
import decimal
import unittest

from uncertainty import binomial_interval,paired_interval


class UncertaintyTest(unittest.TestCase):
    def test_published_and_boundary_examples(self):
        # NIST exact90% example:4 successes among20.
        lo,hi=binomial_interval(4,20,0.90)
        self.assertAlmostEqual(lo,0.071354,delta=5e-7)
        # Published approximate upper root differs by about9e-7. Validate
        # the defining tail separately using Decimal and integer coefficients.
        self.assertAlmostEqual(hi,0.401029,delta=1e-6)
        with decimal.localcontext() as ctx:
            ctx.prec=60
            for k,p,target in [(4,hi,'.05'),(3,lo,'.95')]:
                d=decimal.Decimal(str(p))
                residual=sum(decimal.Decimal(math.comb(20,i))*d**i*(1-d)**(20-i) for i in range(k+1))-decimal.Decimal(target)
                self.assertLess(abs(residual),decimal.Decimal('1e-12'))
        # SciPy official binomtest example, exact95%,7 among50.
        lo,hi=binomial_interval(7,50)
        self.assertAlmostEqual(lo,0.05819170033997342,places=10)
        self.assertAlmostEqual(hi,0.26739600249700846,places=10)
        for n in [1,8,32,500]:
            r=paired_interval(n,0,0)
            expected=-math.expm1(math.log(0.0125)/n)
            self.assertAlmostEqual(r['interval'][0],-expected,places=12)
            self.assertAlmostEqual(r['interval'][1],expected,places=12)
            self.assertGreater(expected,0)
        with self.assertRaises(ValueError):paired_interval(8,5,4)
        with self.assertRaises(ValueError):paired_interval(0,0,0)
        with self.assertRaises(ValueError):paired_interval(8,0,0,0)

    def test_finite_multinomial_grid_and_swap_symmetry(self):
        n=6
        intervals={(w,l):paired_interval(n,w,l)['interval'] for w in range(n+1) for l in range(n-w+1)}
        for (w,l),(lo,hi) in intervals.items():
            mirrored=intervals[(l,w)]
            self.assertAlmostEqual(lo,-mirrored[1],places=12)
            self.assertAlmostEqual(hi,-mirrored[0],places=12)
        for pw in [0,0.05,0.2,0.5,0.8,1]:
            for pl in [0,0.05,0.2,0.5,0.8,1]:
                if pw+pl>1:continue
                coverage=0.0;delta=pw-pl
                for (w,l),(lo,hi) in intervals.items():
                    z=n-w-l
                    probability=math.factorial(n)/(math.factorial(w)*math.factorial(l)*math.factorial(z))*pw**w*pl**l*(1-pw-pl)**z
                    if lo-1e-12<=delta<=hi+1e-12:coverage+=probability
                self.assertGreaterEqual(coverage+1e-12,0.95)


if __name__=='__main__':unittest.main()
