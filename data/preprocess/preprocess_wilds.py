import sys
import os
import Constants
from pathlib import Path 
from data.preprocess.validate import validate_wilds
from data.data_cam import split_train_test

if __name__ == '__main__':
    print("Validating paths...")
    validate_wilds()
    print('Splitting Camelyon...')
    split_train_test(train_ratio=0.8)