import os
from matplotlib import pyplot as plt
import pandas as pd
import gymnasium as gym
import gym_anytrading
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import A2C
from gym_anytrading.envs import StocksEnv
import quantstats as qs

def signals(env):
  start = env.frame_bound[0] - env.window_size
  end = env.frame_bound[1]
  prices = env.df.loc[:, 'close'].to_numpy()[start:end]
  signal_features = env.df.to_numpy()[start:end]
  return prices, signal_features

class CustomEnv(StocksEnv):
  _process_data = signals

class ModelBuilder():
    """
    A class for building, training, and testing an A2C (Advantage Actor-Critic) model using Stable Baselines3.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing relevant data for training and testing.
    - window_size (int): The size of the observation window for the environment.
    - train_percentage (float): The percentage of data to be used for training.
    - eval_callback_freq (int): The frequency of evaluation during training.
    - model_save_path (str): The directory path to save the trained model.

    Attributes:
    - df (pd.DataFrame): The input DataFrame.
    - window_size (int): The size of the observation window for the environment.
    - train_end (int): Index representing the end of the training data based on the specified percentage.
    - model_save_path (str): The directory path to save the trained model.
    - env (stable_baselines3.common.envs.DummyVecEnv): The environment for training the model.
    - stop_callback (stable_baselines3.common.callbacks.StopTrainingOnRewardThreshold): Callback to stop training based on reward threshold.
    - eval_callback (stable_baselines3.common.callbacks.EvalCallback): Callback for model evaluation during training.
    - model (stable_baselines3.model.A2C): The A2C model.

    Methods:
    - train_model(timesteps: int):
        Trains the A2C model for a specified number of timesteps.

        Parameters:
        - timesteps (int): The total number of training timesteps.

    - test_model(frame_bound: Tuple[int, int] = None):
        Tests the trained A2C model on the specified test data.

        Parameters:
        - frame_bound (Tuple[int, int]): The start and end indices of the test data. If not provided,
          the default is set to the reserved test data.

        Example:
        ```python
        # Instantiate ModelBuilder object
        model_builder = ModelBuilder(df=my_dataframe, window_size=10, train_percentage=0.8, eval_callback_freq=1000, model_save_path='models')

        # Train the model
        model_builder.train_model(timesteps=10000)

        # Test the model on the remaining data
        model_builder.test_model()
        ```
    """
    def __init__(self, 
               df: pd.DataFrame, 
               window_size: int, 
               train_percentage: float,
               verbosity = 1):
        self.df = df.copy()
        self.window_size = window_size
        self.train_end = int(train_percentage * len(df))
        env = CustomEnv(df=df, window_size=window_size, frame_bound=(window_size, self.train_end))
        env_lambda = lambda: env
        self.env = DummyVecEnv([env_lambda])
        self.model = A2C('MlpPolicy', self.env, verbose=verbosity)
    
    def load_model(self, model_path):
        if os.path.exists(model_path):
          self.model = A2C.load(model_path)
        else:
           print("File path {} does not exist.".format(model_path))

    def train_model(self, timesteps, callback=None):
        """
            Trains the A2C model for a specified number of timesteps.

            Parameters:
            - timesteps (int): The total number of training timesteps.
        """
        self.model.learn(total_timesteps=timesteps, callback=callback)
        # self.model = A2C.load(self.model_save_path + '/best_model')

    def get_model_report(self, frame_bound=None, plot_preds=True, stats=True):
        """
            Tests the trained A2C model on the specified test data.

            Parameters:
            - frame_bound (Tuple[int, int]): The start and end indices of the test data. If not provided,
          the default is set to the reserved test data.
        """
        if frame_bound == None or frame_bound[0] > frame_bound[1]:
          frame_bound = (self.train_end, len(self.df))
        else:
          beg, end = frame_bound[0], frame_bound[1]
          df_len = len(self.df)
          frame_bound = (int(df_len * beg), int(df_len * end))
        test_env = CustomEnv(df=self.df, window_size=self.window_size, frame_bound=frame_bound)
        obs = test_env.reset()[0]

        while True:
            action, _states = self.model.predict(obs)
            obs, rewards, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            if done:
                print("info", info)
                break
        if plot_preds:
          plt.figure(figsize=(15,6), facecolor='w')
          plt.cla()
          test_env.render_all()
          plt.show()
        
        if stats:
          profits = pd.Series(test_env.history['total_profit'], index=self.df.index[frame_bound[0]+1:])
          returns = profits.pct_change().iloc[1:]
          qs.reports.full(returns)

        return test_env
      