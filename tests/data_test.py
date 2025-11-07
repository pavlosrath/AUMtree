import os
import unittest
import numpy as np
from src.data import load_dataset, ALL_DATASETS, GENERATOR_MAP, DOWNLOADER_MAP


class TestData(unittest.TestCase):

    def check_dataset(self, X: np.ndarray, y: np.ndarray, name: str):
        self.assertIsInstance(
            X, np.ndarray, 
            f"X should be a numpy array in {name}"
        )
        self.assertIsInstance(
            y, np.ndarray, 
            f"y should be a numpy array in {name}"
        )
        self.assertGreater(
            len(X), 0, 
            f"Empty dataset {name}"
        )
        self.assertGreater(
            len(X), 0, 
            f"Empty dataset {name}"
        )
        self.assertTrue(
            len(X) == len(y), 
            f"X and y should have the same length in {name}"
        )
        self.assertTrue(
            y.dtype == int, 
            f"Labels should be integers in {name}"
        )
        self.assertTrue(
            y.min() == 0, 
            f"Labels should start at 0, found {y.min()} in {name}"
        )

    def test_load_datasets(self):
        for name in ALL_DATASETS:
            X, y = load_dataset(name)
            self.check_dataset(X, y, name)

    def test_save_datasets(self):
        save_dir = "./tmp"
        for name in set(ALL_DATASETS) - set(list(GENERATOR_MAP.keys())):
            load_dataset(name, save_dir)
            self.assertTrue(
                os.path.exists(f"{save_dir}/{name}.csv"), 
                f"File not saved for {name}"
            )

            X, y = load_dataset(name, save_dir)
            self.check_dataset(X, y, name)     
            os.remove(f"{save_dir}/{name}.csv")    
            os.rmdir(save_dir)   

if __name__ == "__main__":
    unittest.main()