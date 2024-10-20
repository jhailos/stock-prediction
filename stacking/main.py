from StackingModel import StackingModel
import concurrent.futures
import time
import requests
import pandas as pd
import datetime
import yaml
import json
import os

def download_data(ticker):
    url = "https://api.finazon.io/latest/finazon/us_stocks_essential/time_series"
    ls = []
    for i in range(30):
        querystring = {"ticker":f"{ticker}","interval":"5m","page":f"{i}","page_size":"1000","adjust":"all"}
        with open ("api_key.yml", "r") as file:
            api_key = yaml.safe_load(file)
            headers = {"Authorization": api_key["api_key"]}
        
        while True:
            response = requests.get(url, headers=headers, params=querystring)
            if response.status_code == 429:
                # raise RuntimeError('Used all calls given: wait a minute')
                print(f"waiting 1 second. {i}/30")
                time.sleep(1)
            else:
                break
        data = response.json()

        pandas_data = [{"Date": datetime.datetime.utcfromtimestamp(x["t"]),
                        "Open": x["o"],
                        "High": x["h"],
                        "Low": x["l"],
                        "Close": x["c"],
                        "Volume": x["v"]
                        } for x in data["data"]]
        for item in pandas_data:
            ls.append(item)

    df = pd.DataFrame(ls)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    outdir = "stock_data"
    outfile = f"{ticker}.csv"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    df.to_csv(os.path.join(outdir, outfile))

def get_csv(ticker):
    df = pd.read_csv(f'stock_data\{ticker}.csv', index_col='Date', parse_dates=True)
    return df

def main():
    start_time = time.time()
    ticker = 'AAPL'
    # download_data(ticker)
    stock_data = get_csv(ticker)
    model = StackingModel(ticker, "5m", data=stock_data)
    model.run()

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