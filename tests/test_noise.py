import numpy as np
import unittest

import noise3d
from noise3d import noise

class TestGenseq(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.T = 100
        cls.V = 100
        cls.H = 100
        vals = np.random.normal(0, 1, cls.T*cls.V*cls.H)
        cls.seq = np.reshape(vals, (cls.T,cls.V,cls.H))
        
    def test_10_basic_variances(cls):
        
        var_t = noise3d.noise.var_nt(cls.seq)
        var_v = noise3d.noise.var_nv(cls.seq)
        var_h = noise3d.noise.var_nh(cls.seq)
        var_tv = noise3d.noise.var_ntv(cls.seq)
        var_th = noise3d.noise.var_nth(cls.seq)
        var_vh = noise3d.noise.var_nvh(cls.seq)
        var_tvh = noise3d.noise.var_ntvh(cls.seq)
        singles = (var_t, var_v, var_h, 
                  var_tv, var_th, var_vh,
                  var_tvh)
        
        var_set = noise3d.noise.get_all_3d_noise_var(cls.seq, names=True)
        names = var_set[-1]
        var_set = var_set[:-1]
        expected_names = ('t', 'v', 'h', 
                         'tv', 'th', 'vh',
                         'tvh')
        
        for single, var, name, exp_name in zip(singles, var_set, names, expected_names):
            cls.assertEqual(single, var)
            cls.assertEqual(name, exp_name)
        
    def test_20_fast_equivalent(cls):
        fast_vars = noise3d.noise.get_all_3d_noise_var_fast(cls.seq, names=True)
        classic_vars = noise3d.noise.get_all_3d_noise_var(cls.seq, names=True)
        
        for fast, classic in zip(fast_vars, classic_vars):
            cls.assertAlmostEqual(fast, classic, places=4)
            # cls.assertEqual(fast, classic)
            # cls.assertAlmostEqual(fast, classic)
            
    def test_30_classic_matrix(cls):
        classic_approach = noise3d.noise.get_all_3d_noise_var(cls.seq, names=True)
        matrix_approach = noise3d.noise.get_all_3d_classic_var_matrix(cls.seq, names=True)
        
        for classic, matrix in zip(classic_approach, matrix_approach):
            #cls.assertEqual(classic, matrix)
            cls.assertAlmostEqual(classic, matrix, places=4)
            
            
    def test_40_corrected_matrix(cls):
        corrected = noise3d.noise.get_all_3d_corrected_var_matrix(cls.seq)
        
        
if __name__ == "__main__":
    unittest.main()
