import kagglehub
from dotenv import load_dotenv
from pathlib import Path
import os
import pandas as pd

class Data():
            
    def __init__(self, path_to_data = "", train_val_split = 1800):
        self.data_loader(path_to_data)
        self.train_val_split = train_val_split

    def data_loader(self, path_to_data):
        print("Loading data...")
        # Get Absolute Path of this file 
        BASE_DIR = Path(__file__).resolve().parent 
        self.data_path = path_to_data if path_to_data else f"{BASE_DIR}/../data"

        # By default reads environment variables from ".env" only
        load_dotenv(dotenv_path=".env") 

        # Load the kaggle token
        os.environ["KAGGLE_API_TOKEN"] = os.getenv("KAGGLE_API_TOKEN") 

        if not os.path.exists(f"{self.data_path}/sales_train_evaluation.csv"):
            # Download the actual m5 dataset directly from kaggle (You must have joined this compitition to be able to download (Else will get 403 error))
            kagglehub.competition_download("m5-forecasting-accuracy", output_dir=self.data_path) 

        print("Data loaded succesfully!")

    ''' 
    Various ways train data may be needed:
    Will implement the following 3:
        1. 1 Product -> all it's daywise sales data.
        2. 1 Product -> a range of products with both the daywise sales and prcie data:
            a. data of that product across all the shops in all states
            b. data of all the products in that shop.
    '''
    def get_train_data(self, product_id, choice = ['single_prod_day_wise', 'all_prod_of_1_shop', 'one_prod_in_all_shops']):
        sales = pd.read_csv(f"{self.data_path}/sales_train_validation.csv")
        calendar = pd.read_csv(f"{self.data_path}/calendar.csv")
        sell_prices = pd.read_csv(f"{self.data_path}/sell_prices.csv")

        if choice == 'single_prod_day_wise':
            # data = sales[sales['id'] == product_id][6 : self.train_val_split]
            data = sales[sales['id'] == product_id].iloc[: , 6 : ]

        elif choice == 'all_prod_of_1_shop':
            pass

        elif choice == 'one_prod_in_all_shops':
            pass

        return data
        

    