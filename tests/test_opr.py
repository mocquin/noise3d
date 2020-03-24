import numpy as np
import unittest

from noise3d import opr

class TestGenseq(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.T = 100
        cls.V = 100
        cls.H = 100
        vals = np.random.normal(0, 1, cls.T*cls.V*cls.H)
        cls.seq = np.reshape(vals, (cls.T,cls.V,cls.H))
            
    def test_10_1d_seqs(cls):

if __name__ == "__main__":
    unittest.main()
        
        