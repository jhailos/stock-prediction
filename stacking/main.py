from StackingModel import StackingModel
import concurrent.futures
import time
import requests
import pandas as pd
import datetime

def get_data():
    url = "https://api.finazon.io/latest/finazon/us_stocks_essential/time_series"
    
    querystring = {"ticker":"AAPL","interval":"5m","page":"5","page_size":"1000","adjust":"all"}
    
    headers = {"Authorization": "apikey"}
    
    response = requests.get(url, headers=headers, params=querystring)
    
    print(len(response.json()))
    print(type(response))

def main():
    get_data()
    # start_time = time.time()
    # model = StackingModel("NDX", "5m")
    # model.run()

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
    
    # end_time = time.time()
    # print("Time taken: ", end_time-start_time)

if __name__ == "__main__":
    main()