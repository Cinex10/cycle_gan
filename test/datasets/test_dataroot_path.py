import sys
sys.path.append('/home/cinex/repo/cycle_gan')


import unittest
from data.unpaired_dataset import get_dataroot_path


class TestUnpairedDataset(unittest.TestCase):
    
    def test_dataroot_path(self):
        dataroot = get_dataroot_path(root='datasets/edges2shoes')
        
        self.assertEqual(dataroot,'/home/cinex/repo/cycle_gan/datasets/edges2shoes','data root is wrong')
        

if __name__ == '__main__':
    unittest.main()