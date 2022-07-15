#!usr/bin/env/python3 

import os
import argparse

# Get Arguments
def get_arguments():
    
    parser = argparse.ArgumentParser(
        prog='main',
        description='DDPG Main Node')
    parser.add_argument('-env', '--enviroment_name', type=str,  dest='enviroment_name', default='TrajectoryNavigationMir100Sim-v0', help='Enviroment Name')
    parser.add_argument('-it',  '--is_training',     type=bool, dest='is_training',     default=True,  help='Boolean to switch from Training to Testing')
    parser.add_argument('-ct',  '--custom_testing',  type=bool, dest='custom_testing',  default=False, help='Boolean to Enable Custom Testing')
    parser.add_argument('-num', '--episodes_number', type=int,  dest='episodes_number', default=100,   help='Number of Episodes')
    args = parser.parse_args()
    
    print(f'\nEnviroment Name: {args.enviroment_name} \
            \nTraining: {args.is_training} \
            \nCustom Testing: {args.custom_testing} \
            \nEpisodes Number: {args.episodes_number}\n')

    # Check Enviroment Name    
    if not args.enviroment_name in ['TrajectoryNavigationMir100Sim-v0', 'NoObstacleNavigationMir100Sim-v0']:
        print(f'Enviroment "{args.enviroment_name}" Unavailable | Exit...')
        exit()
    
    return args.enviroment_name, args.is_training, args.custom_testing, args.episodes_number

def remove_pycache(path):
    
    # Remove __pycache__
    if os.path.exists(path):
        
        # Remove Files in Path
        for file in os.listdir(path): os.remove(f'{path}/{file}')
        
        # Remove Path
        os.rmdir(path)

# Create a vector of {lenght} elements with value {values}
def array_initialization(values = 0.0, lenght = 1):
    
    array = [values] * lenght
    return array
