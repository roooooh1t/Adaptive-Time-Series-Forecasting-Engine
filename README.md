# DATA
We are using Walmart m5 dataset for our usecase. We have downloaded the data from the kaggle: 

```https://www.kaggle.com/competitions/m5-forecasting-accuracy/data```

It contains 3 tables:
1. sales : This table contains day-wise sale data for every product (across 3 states in various stores) for about 1900 days.
2. sell_prices : Contains weekly data for price of each product.
3. calendar: It is the link between the 2 tables. the day (in sales) and the week (in sell_prices). Along with this, it contains information about special events which can significantly affect sale of certain products.

To run this, 1st create a ``` .env ``` file which should contain a KAGGLE_API_TOKEN (which you can generate from here: ```https://www.kaggle.com/settings/api```)
```
KAGGLE_API_TOKEN=KGAT_...
```

Then go to ``` /src/data/ ``` and run ```data_loader.py``` to get a local copy of the data. 
For this, first join the competition here : ```https://www.kaggle.com/competitions/m5-forecasting-accuracy```, otherwise you will get 403 (forbidden) errors.


## Differencing Order 
Choosing d = 0, we get 1573 data points (5%) which are non stationary under p95
Choosing d = 2, we get only 1 data point which is non stationary and that too with probability of 0.07 of it being non stationary, Hence we chose d = 2.