import os
import kagglehub
from typing import Literal
from pathlib import Path
from dotenv import load_dotenv

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
    def get_train_data(self, product_id, choice1 : Literal['train', 'validate', 'test'] = 'train' ,  choice2  : Literal['single_prod_day_wise', 'all_prod_of_1_shop', 'one_prod_in_all_shops'] = 'single_prod_day_wise'):
        train_sales = pd.read_csv(f"{self.data_path}/sales_train_validation.csv")
        test_sales = pd.read_csv(f"{self.data_path}/sales_train_evaluation.csv")
        calendar = pd.read_csv(f"{self.data_path}/calendar.csv")
        sell_prices = pd.read_csv(f"{self.data_path}/sell_prices.csv")

        if choice2 == 'single_prod_day_wise':
            # Find the week when this products sale begun
            broken_id = product_id.split('_')
            item_id = broken_id[0] + '_' + broken_id[1] + '_' + broken_id[2]
            store_id = broken_id[3] + '_' + broken_id[4]
            print(item_id, store_id)
            product_wm_yr_wk = sell_prices.loc[(sell_prices['store_id'] == store_id) & (sell_prices['item_id'] == item_id)]['wm_yr_wk'].iloc[0]

            # Find the day from which it's sale would have begun [Assumption: Sale started the very first day of the week it has a sale_price]
            start_day = calendar.index[calendar['wm_yr_wk'] == product_wm_yr_wk][0] # The day number = the row number

            # Get the correct data for a single product
            if choice1 == 'train':
                data = train_sales[train_sales['id'] == product_id].iloc[ : , start_day + 5 : self.train_val_split + 6] ## + 5 and 6 are present to shift the indices appropriately since the day columns start from the 6th column
            elif choice1 == 'validation':
                data = train_sales[train_sales['id'] == product_id].iloc[ : , start_day + 5 : ] ## + 5 is present to shift the indices appropriately since the day columns start from the 6th column
            elif choice1 == 'test':
                data = test_sales[test_sales['id'] == product_id].iloc[ : , start_day + 5 : ] ## + 5 is present to shift the indices appropriately since the day columns start from the 6th column

        elif choice2 == 'all_prod_of_1_shop':
            # Find the store_id and the week it started selling products
            broken_id = product_id.split('_')
            store_id = broken_id[3] + '_' + broken_id[4]
            product_wm_yr_wk = sell_prices.loc[sell_prices['store_id'] == store_id]['wm_yr_wk'].min()

            # Find the day from which it's sale would have begun [Assumption: Sale started the very first day of the week it has a sale_price]
            start_day = calendar.index[calendar['wm_yr_wk'] == product_wm_yr_wk][0] # The day number = the row number

            # Get the correct data for a single product
            if choice1 == 'train':
                data = train_sales[train_sales['store_id'] == store_id].iloc[ : , start_day + 5 : self.train_val_split + 6] ## + 5 and 6 are present to shift the indices appropriately since the day columns start from the 6th column
            
            elif choice1 == 'validation':
                data = train_sales[train_sales['store_id'] == store_id].iloc[ : , start_day + 5 : ] ## + 5 is present to shift the indices appropriately since the day columns start from the 6th column
            
            elif choice1 == 'test':
                data = test_sales[test_sales['store_id'] == store_id].iloc[ : , start_day + 5 : ] ## + 5 is present to shift the indices appropriately since the day columns start from the 6th column
            

        elif choice2 == 'one_prod_in_all_shops':
            # Find the product_id and the week this product's sale started in any of the stores.
            broken_id = product_id.split('_')
            item_id = broken_id[0] + '_' + broken_id[1] + '_' + broken_id[2]
            product_wm_yr_wk = sell_prices.loc[sell_prices['item_id'] == item_id]['wm_yr_wk'].min()

            # Find the day from which it's sale would have begun [Assumption: Sale started the very first day of the week it has a sale_price]
            start_day = calendar.index[calendar['wm_yr_wk'] == product_wm_yr_wk][0] # The day number = the row number

            # Get the correct data for a single product
            if choice1 == 'train':
                data = train_sales[train_sales['item_id'] == item_id].iloc[ : , start_day + 5 : self.train_val_split + 6] ## + 5 and 6 are present to shift the indices appropriately since the day columns start from the 6th column
            
            if choice1 == 'validation':
                data = train_sales[train_sales['item_id'] == item_id].iloc[ : , start_day + 5 : ] ## + 5 is present to shift the indices appropriately since the day columns start from the 6th column
            
            if choice1 == 'test':
                data = test_sales[test_sales['item_id'] == item_id].iloc[ : , start_day + 5 : ] ## + 5 is present to shift the indices appropriately since the day columns start from the 6th column
            
        return data
        

    