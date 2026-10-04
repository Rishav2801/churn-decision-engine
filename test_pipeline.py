import unittest
import numpy as np
from pipeline import top_indices, preprocess
class ChurnTests(unittest.TestCase):
    def test_capacity_and_order(self):
        np.testing.assert_array_equal(top_indices([.1,.9,.5,.2,.8],.4),[1,4])
    def test_invalid_budget(self):
        with self.assertRaises(ValueError): top_indices([.1],0)
    def test_no_identifier_features(self):
        columns=[c for _,_,cols in preprocess().transformers for c in cols]
        self.assertNotIn('customerID',columns)
        self.assertNotIn('Churn',columns)
if __name__=='__main__': unittest.main()
