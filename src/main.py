#!usr/bin/env/python3

import os, sys, getopt, pathlib, rospy
import numpy as np

# from enviroment import MPO500_Env
from DDPG.agent import Agent
from DDPG.utils import plot_learning_curve
# from utils import remove_pycache as rm_pycache

import gym, robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling

class Program:

    def __init__(self, name='motion_planner_DDPG', env_name='NoObstacleNavigationMir100Sim-v0', is_training=False, custom_testing=False, episodes_number=250):

        # Load Rospy Parameters
        self.is_training = is_training
        self.custom_testing = custom_testing
        self.n_episodes = episodes_number

        # Get Current Directory
        self.current_directory = pathlib.Path(__file__).parent.resolve()

        # Figure File
        if not os.path.exists(f'{self.current_directory}/Plots'): os.makedirs(f'{self.current_directory}/Plots')
        self.figure_file = f'{self.current_directory}/Plots/{name}.png'

        # Create Enviroment
        # self.env = MPO500_Env()
        target_machine_ip = '127.0.0.1' # or other machine 'xxx.xxx.xxx.xxx'

        # initialize environment
        print('\nStarting Enviroment\n')
        self.env_name = env_name
        self.env = gym.make(self.env_name, ip=target_machine_ip, gui=True)
        self.env = ExceptionHandling(self.env)
        # self.env = gym.make('Pendulum-v1')
        print('\nEnviroment Created\n')

        # Score History
        self.best_score = self.env.reward_range[0]
        self.score_history = []

        # Create Agent
        self.agent = Agent(input_dims=self.env.observation_space.shape, env=self.env, n_actions=self.env.action_space.shape[0], 
                           chkpt_dir=f'{self.current_directory}/Models')
        print('\nAgent Created\n')

        # rospy.init_node(name, anonymous=True)
        # rate = rospy.Rate(500)
        
    def train(self):
        
        # For Loop for Running Every Episode
        for i in range(self.n_episodes):

            # Reset the Enviroment at the Beginning on Every Episode
            observation = self.env.reset()
            done = False
            score = 0

            # Until the Episode is Not Done
            while not done: # and not rospy.is_shutdown():

                # Choose and Action | Set Evaluate to False
                action = self.agent.choose_action(observation, evaluate=False)
                
                # Get Observation and Rewards from the Enviroment
                new_observation, reward, done, info = self.env.step(action)
        
                # Increment Score
                score += reward
       
                # Remember the Episode
                self.agent.remember(observation, action, reward, new_observation, done)

                # Learn
                self.agent.learn()

                # Update the Observation
                observation = new_observation

            # Append the Score and Compute the Mean
            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-100:])

            # If the Score of this Episode is Better than the Others
            if avg_score > self.best_score:

                # Update Best Score
                self.best_score = avg_score

                # Save the Model
                self.agent.save_models()

            # Debug Print at the End of Each Episode
            print('episode {:4.0f}\t|\tscore {:10.6f}\t|\tavg score {:10.6f}'.format(i, score, avg_score))
            
        # Compute the X-Axes of the Graph
        x = [i+1 for i in range(self.n_episodes)]

        # Plot the Training Graph
        plot_learning_curve(x, self.score_history, self.figure_file)
    
        
    def random_testing(self):
        
        # Load Agent Model
        self.load_model()
        
        # For Loop for Running Every Episode
        for i in range(self.n_episodes):

            # Reset the Enviroment at the Beginning on Every Episode
            observation = self.env.reset()
            done = False
            score = 0

            # Until the Episode is Not Done
            while not done: # and not rospy.is_shutdown():

                # Choose and Action
                action = self.agent.choose_action(observation, evaluate=True)
                
                # Get Observation and Rewards from the Enviroment
                new_observation, reward, done, info = self.env.step(action)

                # Remember the Episode
                self.agent.remember(observation, action, reward, new_observation, done)

                # Update the Observation
                observation = new_observation

        
    # Custom Scenario Mir100Env
    def Mir100_testing(self, starting_pose=[0.0,0.0,0.0], target_pose=[1.0,1.0,0.0]):
        
        # Load Agent Model
        self.load_model()
        
        # Reset the Enviroment setting Custom Starting Pose and Target Pose
        observation = self.env.reset(start_pose=starting_pose, target_pose=target_pose)
        done = False

        # Until the Episode is Not Done
        while not done: # and not rospy.is_shutdown():

            # Choose and Action
            action = self.agent.choose_action(observation, evaluate=True)
            
            # Get Observation and Rewards from the Enviroment
            new_observation, reward, done, info = self.env.step(action)

            # Remember the Episode
            self.agent.remember(observation, action, reward, new_observation, done)

            # Update the Observation
            observation = new_observation


    def load_model(self):
        
        n_steps = 0

        # In order to load we have to fill up the memory with dummy values calling the lean function
        while n_steps <= self.agent.batch_size: # and not rospy.is_shutdown():

            # Do Observation and get Random Actions
            observation = self.env.reset()
            action = self.env.action_space.sample()
            new_observation, reward, done, info = self.env.step(action)
            self.agent.remember(observation, action, reward, new_observation, done)
            n_steps += 1

        # Call the Dummy Learning
        self.agent.learn()

        # Load Models
        self.agent.load_models()
        

    # def remove_pycache(self):
        
        # # Remove __pycache__
        # rm_pycache(f'{self.current_directory}/__pycache__')
        # rm_pycache(f'{self.current_directory}/DDPG/__pycache__')
        # rm_pycache(f'{self.current_directory}/MPO500/__pycache__')

# Get Arguments
def getargv(argv):
    
    enviroment_name = 'TrajectoryNavigationMir100Sim-v0'
    is_training = True
    custom_testing = False
    episodes_number = 100
    help = "\n{0} -env <enviroment_name> -it <is_training> -ct <custom_testing -num <episodes_number>\n".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "h:env:it:ct:num:", ["help", "enviroment_name=", "is_training=", "custom_testing=", "episodes_number"])
    except:
        print(help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help)  # print the help message
            sys.exit(2)
        elif opt in ("-env", "--enviroment_name"):
            enviroment_name = arg
        elif opt in ("-it", "--is_training"):
            is_training = arg
        elif opt in ("-ct", "--custom_testing"):
            custom_testing = arg
        elif opt in ("-num", "--episodes_number"):
            episodes_number = arg
            
    print(f'\nEnviroment Name: {enviroment_name} \
            \nTraining: {is_training} \
            \nCustom Testing: {custom_testing} \
            \nEpisodes Number: {episodes_number}\n')

    # Check Enviroment Name    
    if not enviroment_name in ['TrajectoryNavigationMir100Sim-v0', 'NoObstacleNavigationMir100Sim-v0']:
        print(f'Enviroment "{enviroment_name}" Unavailable | Exit...')
        exit()
        
    return enviroment_name, is_training, custom_testing, episodes_number
        
if __name__ == "__main__":
    
    # Get Arguments
    enviroment_name, is_training, custom_testing, episodes_number = getargv(sys.argv)
    
    # DDPG = Program('motion_planner_DDPG', NoObstacleNavigationMir100Sim-v0, is_training=True,  custom_testing=False, episodes_number=100)
    # DDPG = Program('motion_planner_DDPG', NoObstacleNavigationMir100Sim-v0, is_training=False, custom_testing=False, episodes_number=100)
    # DDPG = Program('motion_planner_DDPG', NoObstacleNavigationMir100Sim-v0, is_training=False,  custom_testing=True, episodes_number=100)
    
    # DDPG = Program('motion_planner_DDPG', TrajectoryNavigationMir100Sim-v0, is_training=True,  custom_testing=False, episodes_number=100)
    
    DDPG = Program('motion_planner_DDPG', enviroment_name, is_training,  custom_testing, episodes_number)

    # Train Model
    if DDPG.is_training: DDPG.train()

    # Rndom Testing Model
    elif not DDPG.custom_testing: DDPG.random_testing()
    
    # Custom Testing Model
    elif DDPG.custom_testing and DDPG.env_name == 'NoObstacleNavigationMir100Sim-v0': 
        
        # starting_pose = [x,y,yaw] | -2.0 < x,y < 2.0 | -pi < yaw < pi
        starting_pose = [0.0, 0.0, 0.0]
        
        # target_pose   = [x,y,yaw] | -1.0 < x,y < 1.0 | yaw = 0.0
        target_pose = [10.0, -5.0, 0.0]
        
        DDPG.Mir100_testing(starting_pose, target_pose)

    # Remove __pycache__
    DDPG.remove_pycache()
