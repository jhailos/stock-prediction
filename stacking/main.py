from StackingModel import StackingModel
from ContextData import ContextData
from StrategyYfinance import StrategyYfinance
from StrategyFinazon import StrategyFinazon
from StrategyAlphaVantage import StrategyAlphaVantage
from StrategyAlpaca import StrategyAlpaca
import seaborn as sns
import matplotlib.pyplot as plt

import time

def main():
    print('>Fetching data')
    context = ContextData(ticker="SPY", strategy=StrategyAlpaca(), days=10, interval="1m", market_hours_only=False)
    context.write_csv()
    start_time = time.time()
    
    plt.figure(figsize=(12, 6))
    plt.plot(context.data['Close'], label=f'{context.ticker} Closing Price', linewidth=2)
    plt.title(f'{context.ticker} Closing Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.legend()
    plt.show()

    sns.displot(data=context.data, x="price_change", kde=True)
    plt.show()

    model = StackingModel(context.data, context.interval)
    rmse, rrmse, r2, predicted_price, last_price = model.run()

    print('---------------------------')
    lower_bound = predicted_price - rmse
    upper_bound = predicted_price + rmse
    print('LAST PRICE: ', last_price)
    print("> Upper bound: ", upper_bound)
    print("> Actual: ", predicted_price)
    print("> Lower bound: ", lower_bound)
    if last_price < lower_bound:
        print(f'# Going up by {predicted_price - last_price}')
    elif last_price > upper_bound:
        print(f'# Going down by {last_price - predicted_price}')
    else:
        print('# RMSE too high to make accurate prediction')

    end_time = time.time()
    print('---------------------------')
    print("Time taken: ", end_time-start_time)

if __name__ == "__main__":
    main()