import numpy as np
import unittest

from noise3d import bias

class TestGenseq(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.T = 100
        cls.V = 100
        cls.H = 100
        cls.var = 1
        
    def test_00_error_functions(cls):
        funcs = (bias.var_t_err_tvh, bias.var_v_err_tvh, bias.var_h_err_tvh,
                 bias.var_tv_err_tvh, bias.var_th_err_tvh, bias.var_vh_err_tvh, 
                 bias.var_tvh_err_tvh)
        for func in funcs:
            try:
                func(cls.T, cls.V, cls.H, cls.var)
            except:
                cls.fail()
        
    def test_10_demo(cls):
        demo = bias.BiasedEstimatorTVH(V=10, H=10, nb_matrix=20)
        try:
            demo.setup()
            demo.compute_results()
        except:
            cls.fail()
        

if __name__ == "__main__":
    unittest.main()
