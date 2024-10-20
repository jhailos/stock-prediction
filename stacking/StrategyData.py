from abc import ABC, abstractmethod

class StrategyData(ABC):

    @abstractmethod
    def download_data(self, ticker, days, interval):
        pass