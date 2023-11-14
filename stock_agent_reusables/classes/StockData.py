from yahoo_fin.stock_info import get_data
import pandas as pd
from typing import List, Tuple, Callable
from stock_agent_reusables.utils import apply_scaler_to_data, get_scaler_for_data

class StockData():
  """
    A class to retrieve and manipulate stock data using Yahoo Finance API.

    Parameters:
    - ticker (str): The stock symbol.
    - start_date (str): The start date for fetching historical data (format: 'MM/DD/YYYY').
    - end_date (str): The end date for fetching historical data (format: 'MM/DD/YYYY').
    - interval (str): The frequency of data points ('1d', '1wk', '1mo', or '1m').

    Attributes:
    - ticker (str): The stock symbol.
    - start_date (str): The start date for fetching historical data.
    - end_date (str): The end date for fetching historical data.
    - interval (str): The frequency of data points.
    - stock_df (pd.DataFrame): DataFrame containing retrieved stock data.

    Methods:
    - add_indicators_to_data(indicators: List[Callable[[pd.DataFrame], Tuple[List, str]]]) -> pd.DataFrame:
        Adds user-defined indicators to the stock data DataFrame.

        Parameters:
        - indicators (List[Callable[[pd.DataFrame], Tuple[List, str]]]): A list of indicator functions.
          Each function should take a DataFrame as input and return a tuple containing the indicator data
          (list-like) and the name of the indicator column (str). The passed-in DataFrame is guaranteed to have
        'close', 'volume', 'low', and 'high' columns.

        Returns:
        - pd.DataFrame: Altered DataFrame with added indicator columns.

    Example:
    ```python
    # Instantiate StockData object
    stock_data = StockData(ticker='AAPL', start_date='01/01/2021', end_date='01/01/2023', interval='1d')

    # Define custom indicator function
    def custom_indicator(data: pd.DataFrame) -> Tuple[List, str]:
        # ... (implement custom logic)
        return indicator_data, 'Custom_Indicator'

    # Add custom indicator to stock data
    modified_data = stock_data.add_indicators_to_data(indicators=[custom_indicator])
    ```

    Note:
    - This class uses the Yahoo Finance API through the `get_data` function to retrieve stock data.
    - The `add_indicators_to_data` method allows users to enhance the stock data DataFrame by adding custom indicators.
  """
  def __init__(self,
              ticker: str,
              start_date: str,
              end_date: str,
              interval: str):
    self.ticker = ticker
    self.start_date = start_date
    self.end_date = end_date
    self.interval = interval
    stock_df = get_data(ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        index_as_date=True,
                        interval=interval)
    self.stock_df = stock_df.drop("ticker", axis=1)
    self.scaler = None

  def add_indicators_to_data(self, indicators: List[Callable[[pd.DataFrame], Tuple[List, str]]]) -> pd.DataFrame:
    """
      Adds user-defined indicators to the stock data DataFrame.

      Parameters:
      - indicators (List[Callable[[pd.DataFrame], Tuple[List, str]]]): A list of indicator functions.
        Each function should take a DataFrame as input and return a tuple containing the indicator data
        (list-like) and the name of the indicator column (str). The passed-in DataFrame is guaranteed to have
        'close', 'volume', 'low', and 'high' columns.

      Returns:
      - pd.DataFrame: Altered DataFrame with added indicator columns.
    """
    df_copy = self.stock_df.copy()
    for indicator in indicators:
      col_data, col_name = indicator(df_copy)
      if len(col_data) == len(df_copy):
        df_copy[col_name] = col_data
    df_copy.dropna(inplace=True)
    return df_copy

  def scale_data(self, data: pd.DataFrame, columns_to_scale: List[str], inverse: bool) -> pd.DataFrame:
    data_copy = data.copy()
    if self.scaler == None:
      self.scaler = get_scaler_for_data(data_copy['close'][:])
    
    for column in columns_to_scale:
      data_copy[column] = apply_scaler_to_data(data[column], self.scaler, inverse)
    
    return data_copy
    
  def get_classification_labels(self, hold_threshold_percentage=0) -> List[int]:
    labels = [0]
    closes = self.stock_df['close']
    for i in range(1, len(closes)):
      percent_change = closes[i] / closes[i-1]
      if percent_change > 1:
        percent_change -= 1
        if percent_change < hold_threshold_percentage:
          labels.append(2)
        else:
          labels.append(1)
      else:
        percent_change = 1 - percent_change
        if percent_change < hold_threshold_percentage:
          labels.append(2)
        else:
          labels.append(0)
    return labels
        
