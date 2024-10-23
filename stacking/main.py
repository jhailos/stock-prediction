from StackingModel import StackingModel
from ContextData import ContextData
from StrategyYfinance import StrategyYfinance
from StrategyFinazon import StrategyFinazon

import concurrent.futures
import time
import requests
import pandas as pd
import datetime
import yaml
import json
import os

def main():
    start_time = time.time()
    context = ContextData(ticker="^GSPC", strategy=StrategyYfinance(), days=59, interval="2m")
    model = StackingModel(context.data, context.interval)
    rmse, rrmse, r2, predicted_price, last_price = model.run()

    print('---------------------------')
    lower_bound = predicted_price - rmse
    upper_bound = predicted_price + rmse
    print("> Upper bound: ", upper_bound)
    print("> Actual: ", predicted_price)
    print("> Lower bound: ", lower_bound)
    if predicted_price < lower_bound:
        print(f'# Going down {last_price - predicted_price}')
    elif predicted_price > upper_bound:
        print(f'# Going up by {predicted_price - last_price}')
    else:
        print('# RMSE too high to make accurate prediction')


    # # multiprocessing (concurrent.futures)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     results = [executor.submit(model.run) for _ in range (8)]

    #     for f in concurrent.futures.as_completed(results):
    #         pass
    #         # print(f.result())

    # # multiprocessing
    # processes = []
    # for _ in range(20):
    #     p = mp.Process(target=model.run)
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
    
    end_time = time.time()
    print("Time taken: ", end_time-start_time)

if __name__ == "__main__":
    main()