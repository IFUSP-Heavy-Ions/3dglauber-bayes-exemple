"""Run with python -m unittest test_experimental_cs_covariance."""
import ast
from pathlib import Path
import unittest
import numpy as np
from scipy.linalg import lapack
from scipy.stats import multivariate_normal
from experimental_cs_covariance import compound_symmetry_covariance as cs


class CSTests(unittest.TestCase):
    def test_endpoints_and_negative_bound(self):
        std=np.array([1.,2.,3.])
        np.testing.assert_array_equal(cs(std,1),np.outer(std,std))
        np.testing.assert_array_equal(cs(std,0),np.diag(std**2))
        self.assertGreaterEqual(np.linalg.eigvalsh(cs(std,-.5)).min(),-1e-12)
        with self.assertRaises(ValueError):cs(std,-1)

    def test_heterogeneous_observables_and_total_likelihood(self):
        # Extract pure functions without importing emulators or launching preprocessing.
        tree=ast.parse(Path(__file__).with_name('bayes_mcmc_cs_model.py').read_text())
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and
                   n.name in ['compute_cov','mvn_loglike','normed_mvn_loglike']]
        ns=dict(np=np,lapack=lapack,compound_symmetry_covariance=cs,transform_multiplicities=True)
        exec(compile(ast.Module(body=functions,type_ignores=[]),'<tests>','exec'),ns)
        std=np.array([.2,.4,.6,.8]);exp=cs(std,1)
        self.assertEqual(np.linalg.matrix_rank(exp),1)
        total=exp+np.diag([.03,.04,.05,.06]);y=np.array([.1,-.2,.3,-.1])
        expected=multivariate_normal.logpdf(y,cov=total)
        self.assertAlmostEqual(ns['normed_mvn_loglike'](y,total.copy()),expected)
        self.assertAlmostEqual(ns['mvn_loglike'](y,total.copy()),expected+len(y)/2*np.log(2*np.pi))
        with self.assertRaises(np.linalg.LinAlgError):ns['mvn_loglike'](y,np.zeros((4,4)))
        dy=np.array([.1,.2]);log_std=np.sqrt(np.log1p(dy**2))
        np.testing.assert_allclose(ns['compute_cov']('dNdeta','dNdeta',dy,dy),np.outer(log_std,log_std))
        np.testing.assert_allclose(ns['compute_cov']('dNdeta','v22',dy,dy),np.outer(log_std,dy))


if __name__=='__main__':unittest.main()
