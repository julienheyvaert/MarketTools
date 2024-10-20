import pandas as pd
import matplotlib.pyplot as plt
from data_fetcher import *
from indicators import *
import os

class PlotManager: 
    def __init__(self, 
                 title: str = None,
                 x_label: str = None,
                 y_label: str = None,
                 legend: bool = True,
                 grid: bool = True):
        
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.legend = legend
        self.grid = grid
        self.plots = []

    def add_plot(self, line_data: pd.Series, plot_type: str = 'line', 
                 color: str = 'orange', label: str = None, alpha: float = 1, scatter_size = None, scatter_marker = None, vertical_style = ':'):
        if not isinstance(line_data, pd.Series):
            raise ValueError('Error: line_data must be a pandas.Series object.')

        self.plots.append((line_data, plot_type, color, label, alpha, scatter_size, scatter_marker, vertical_style))

    def cumulate_plots(self):
        plt.figure(figsize=(12, 6))

        for line_data, plot_type, color, label, alpha, scatter_size, scatter_marker, vertical_style in self.plots:
            if plot_type == 'line':
                plt.plot(line_data.index, line_data, label=label, color=color, alpha=alpha)
            elif plot_type == 'scatter':
                plt.scatter(line_data.index, line_data, label=label, color=color, alpha=alpha, s = scatter_size, marker = scatter_marker)
            elif plot_type == 'vertical_line':
                for date in line_data.index:
                    plt.axvline(x=date, color=color, label=label, alpha=alpha, ls = vertical_style)

        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        if self.legend:
            plt.legend()
        if self.grid:
            plt.grid()

    def show(self):
        self.cumulate_plots()
        plt.show()
    
    def save(self, filename: str):
        directory = "plots"
        if not os.path.exists(directory):
            os.makedirs(directory)

        filepath = os.path.join(directory, filename)
        self.cumulate_plots()

        plt.savefig(filepath)

if __name__ == "__main__":
    ticker1 = 'BTC-USD'
    start_date = '2024-09-30'
    market_data = fetch_market_data(ticker1, start_date)

    plot_manager = PlotManager(title='TA Plot', x_label='Dates', y_label='Prices')
    plot_manager.add_plot(market_data['Close'], color='blue', label='Price', alpha=0.5)

    def plot_extremums(series = market_data['Close']):
        ath_atl_series = find_extremums(series)    
        ath_series = ath_atl_series[ath_atl_series['extremum_type'] == 1]['extremum_value']
        atl_series = ath_atl_series[ath_atl_series['extremum_type'] == -1]['extremum_value']

        plot_manager.add_plot(ath_series, plot_type='scatter', color='orange', label='Local ATH', alpha=0.7)
        plot_manager.add_plot(atl_series, plot_type='scatter', color='purple', label='Local ATH', alpha=0.7)
    
    def plot_psar(series_close = market_data['Close'], series_high = market_data['High'], series_low = market_data['Low']):
        if series_close is None or series_close.empty or \
                series_high is None or series_high.empty or \
                series_low is None or series_low.empty:
            raise ValueError('Error: Missing or empty series_close, series_high, or series_low.')

        
        psar_data = psar(series_close, series_high, series_low)
        plot_manager.add_plot(psar_data['psar'], plot_type='scatter', color='grey', label='Parabolic SAR', scatter_size=3)

        psar_bullish = psar_data[psar_data['psar_position'] == -1]['psar']
        plot_manager.add_plot(psar_bullish, plot_type='scatter', color='green', label='Parabolic SAR', scatter_marker='^')
        psar_bearish = psar_data[psar_data['psar_position'] == 1]['psar']
        plot_manager.add_plot(psar_bearish, plot_type='scatter', color='red', label='Parabolic SAR', scatter_marker='v')

        psar_position = psar_data[psar_data['psar_position'] != 0]
        plot_manager.add_plot(psar_position.index.to_series(), plot_type='vertical_line', alpha=0.3, color='grey')

    plot_psar()
    plot_manager.save('cumulated_plot.png')
