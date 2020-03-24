# check for dims and shapes
import numpy as np
import unittest

from noise3d import genseq

class TestGenseq(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.T = 100
        cls.V = 100
        cls.H = 100
            
    def test_10_1d_seqs(cls):
        # checking dims and shapes
        cls.assertEqual(genseq.genseq_t(cls.T, cls.V, cls.H, 0, 1).shape, (cls.T, cls.V, cls.H))
        cls.assertEqual(genseq.genseq_v(cls.T, cls.V, cls.H, 0, 1).shape, (cls.T, cls.V, cls.H))
        cls.assertEqual(genseq.genseq_h(cls.T, cls.V, cls.H, 0, 1).shape, (cls.T, cls.V, cls.H))
        
    def test_20_2d_seqs(cls):
        # checking dims and shapes
        cls.assertEqual(genseq.genseq_tv(cls.T, cls.V, cls.H, 0, 1).shape, (cls.T, cls.V, cls.H))
        cls.assertEqual(genseq.genseq_th(cls.T, cls.V, cls.H, 0, 1).shape, (cls.T, cls.V, cls.H))
        cls.assertEqual(genseq.genseq_vh(cls.T, cls.V, cls.H, 0, 1).shape, (cls.T, cls.V, cls.H))

    def test_30_3d_seq(cls):
        # checking dims and shapes
        cls.assertEqual(genseq.genseq_tvh(cls.T, cls.V, cls.H, 0, 1).shape, (cls.T, cls.V, cls.H))
        
    def test_40_gen_seqs(cls):
        mus = tuple([0 for i in range(7)])
        sigmas = tuple([1 for i in range(7)])
        
        seqs = genseq.genseq_all_seq(cls.T, cls.V, cls.H, sigmas, mus=mus)
        
        # checking dims and shapes
        for seq in seqs:
            cls.assertEqual(seq.shape, (cls.T, cls.V, cls.H))

            
        cls.assertTrue(np.all(seqs[-1] == sum(seqs[:-1])))
        

if __name__ == "__main__":
    unittest.main()
        
        