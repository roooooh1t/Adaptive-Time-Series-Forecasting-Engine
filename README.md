# DATA
We are using Walmart m5 dataset for our usecase. We have downloaded the data from the kaggle: 

```https://www.kaggle.com/competitions/m5-forecasting-accuracy/data```

To run this, 1st create a ``` .env ``` file which should contain a KAGGLE_API_TOKEN (which you can generate from here: ```https://www.kaggle.com/settings/api```)
```
KAGGLE_API_TOKEN=KGAT_...
```

Then go to ``` /src/data/ ``` and run ```data_loader.py``` to get a local copy of the data. 
For this, first join the competition here : ```https://www.kaggle.com/competitions/m5-forecasting-accuracy```, otherwise you will get 403 (forbidden) errors.
