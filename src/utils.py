#!usr/bin/env/python3 

import os

def remove_pycache(path):
    
    # Remove __pycache__
    if os.path.exists(path):
        
        # Remove Files in Path
        for file in os.listdir(path): os.remove(f'{path}/{file}')
        
        # Remove Path
        os.rmdir(path)

def array_initialization(values = 0.0, lenght = 1):
    
    # Create a vector of {lenght} elements with value {values}
    array = [values] * lenght
    return array
