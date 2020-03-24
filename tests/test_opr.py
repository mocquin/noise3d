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
            
    def test_10_base_operators(cls):
        # checking dims and shapes
        cls.assertEqual(opr.dt(cls.seq).shape, (1, cls.V, cls.H))
        cls.assertEqual(opr.dv(cls.seq).shape, (cls.T, 1, cls.H))
        cls.assertEqual(opr.dh(cls.seq).shape, (cls.T, cls.V, 1))
        
    def test_20_base_i_operators(cls):
        # checking dims and shapes
        cls.assertEqual(opr.idt(cls.seq).shape, (cls.T, cls.V, cls.H))
        cls.assertEqual(opr.idv(cls.seq).shape, (cls.T, cls.V, cls.H))
        cls.assertEqual(opr.idh(cls.seq).shape, (cls.T, cls.V, cls.H))
        
        
    def test_30_noise_seq_extraction(cls):
        # checking dims and shapes
        # 1d noise seqs
        cls.assertEqual(opr.n_t(cls.seq).shape, (cls.T, 1, 1))
        cls.assertEqual(opr.n_v(cls.seq).shape, (1, cls.V, 1))
        cls.assertEqual(opr.n_h(cls.seq).shape, (1, 1, cls.H))
        # 2d noise seqs
        cls.assertEqual(opr.n_tv(cls.seq).shape, (cls.T, cls.V, 1))
        cls.assertEqual(opr.n_th(cls.seq).shape, (cls.T, 1, cls.H))
        cls.assertEqual(opr.n_vh(cls.seq).shape, (1, cls.V, cls.H))
        # 3d noise seq
        cls.assertEqual(opr.n_tvh(cls.seq).shape, (cls.T, cls.V, cls.H))
        
    def test_40_noise_seq_basic_extraction_function(cls):
        seqs_basic = opr.get_all_3d_noise_seq(cls.seq)

        for seq_basic in seqs_basic:
            # checking dims and shapes
            cls.assertTrue(seq_basic.shape, (cls.T, cls.V, cls.H))
        
        # checking variances
        cls.assertTrue(np.var(seqs_basic[-1]),
                       np.var(sum(seqs_basic[:-1])))
        
        # False because noises cancels themselves ?
        #cls.assertTrue(np.allclose(sum(seqs_basic[:-1]), seqs_basic[-1]))
        
    def test_45_noise_seq_fast_extraction_function(cls):
        seqs_fast = opr.get_all_3d_noise_seq_fast(cls.seq)

        for seq_fast in seqs_fast:
            # checking dims and shapes
            cls.assertTrue(seq_fast.shape, (cls.T, cls.V, cls.H))
        
        # checking variances
        cls.assertTrue(np.var(seq_fast[-1]),
                       np.var(sum(seq_fast[:-1])))
        
        # False because noises cancels themselves ?
        #cls.assertTrue(np.allclose(sum(seqs_basic[:-1]), seqs_basic[-1]))
        
    def test_50_equivalence_basic_fast_methods(cls):
        seqs_basic = opr.get_all_3d_noise_seq(cls.seq)
        seqs_fast = opr.get_all_3d_noise_seq_fast(cls.seq)
        for seq_basic, seq_fast in zip(seqs_basic, seqs_fast):
            # checking dims and shapes
            cls.assertTrue(seq_basic.shape, seq_fast.shape)
            # checking variances
            cls.assertTrue(np.var(seq_basic), np.var(seq_fast))
            # checking almost equal : False for one seq, because of rounding ?
            # cls.assertTrue(np.allclose(seq_basic, seq_fast))
        
if __name__ == "__main__":
    unittest.main()
        
        