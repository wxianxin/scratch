"""
Here are some additional modules that could be useful to include in a Python backtesting framework:

Risk management: You may want to include features that allow you to control the level of risk in your trades, such as position sizing and stop-loss orders.

Order execution: You may want to include features that simulate the process of executing trades, including the handling of market orders, limit orders, and other types of orders.

Portfolio optimization: You may want to include tools for optimizing the composition of your portfolio, such as by maximizing return or minimizing risk.

Data visualization: You may want to include tools for visualizing the performance of your strategy and portfolio, such as charts and plots.

Reporting: You may want to include features for generating reports on the performance of your strategy and portfolio, such as PDF or Excel documents.

Integration with live data feeds and brokerage APIs: If you want to use your backtesting system to trade live markets, you will need to integrate it with real-time data feeds and brokerage APIs.
"""


import numpy as np
import matplotlib.pyplot as plt


class Backtest:
    """"""

    def __init__(self, data, strategy, capital, risk_manager, visualizer):
        self.data = data
        self.strategy = strategy
        self.capital = capital
        self.initial_capital = capital
        self.positions = []
        self.portfolio = []
        self.risk_manager = risk_manager
        self.visualizer = visualizer

    def run(self):

        for i in range(len(self.data)):
            # Get the current date and prices
            date = self.data.index[i]
            prices = self.data.iloc[i]

            # Check if we need to make any trades
            trade_signal = self.strategy.generate_trade_signal(self.data[: i + 1])
            if trade_signal == "BUY":
                # Calculate the number of shares to buy
                risk = self.strategy.get_risk_percentage(self.data[: i + 1])
                shares = self.risk_manager.get_position_size(prices, risk)

                # Update the capital and positions
                self.capital -= shares * prices["close"]
                self.positions.append(
                    {
                        "date": date,
                        "Type": "BUY",
                        "Shares": shares,
                        "Price": prices["close"],
                    }
                )
            elif trade_signal == "SELL":
                # Find the most recent buy position
                for j in range(len(self.positions) - 1, -1, -1):
                    if self.positions[j]["Type"] == "BUY":
                        break

                # Calculate the profit/loss from the sell
                sell_price = prices["close"]
                buy_price = self.positions[j]["Price"]
                profit_loss = sell_price - buy_price
                shares = self.positions[j]["Shares"]

                # Update the capital and positions
                self.capital += prices["close"] * shares
                self.positions = self.positions[:j]

            # Update the portfolio value
            self.portfolio.append(
                {
                    "date": date,
                    "value": self.capital
                    + sum([p["Shares"] * prices["close"] for p in self.positions]),
                }
            )

    def get_performance(self):
        # Calculate the profit/loss for each trade
        trades = []
        for i in range(len(self.positions)):
            if self.positions[i]["Type"] == "BUY":
                continue
            for j in range(i - 1, -1, -1):
                if self.positions[j]["Type"] == "BUY":
                    break
            sell_price = self.positions[i]["Price"]
            buy_price = self.positions[j]["Price"]
            profit_loss = sell_price - buy_price
            trades.append(profit_loss)

        # Calculate the overall return
        total_return = (
            self.portfolio[-1]["value"] - self.initial_capital
        ) / self.initial_capital

        # Calculate the Sharpe ratio
        sharpe_ratio = np.mean(trades) / np.std(trades)

        # Plot the returns
        returns = [p["value"] / self.initial_capital - 1 for p in self.portfolio]
        self.visualizer.plot_returns(returns, "Strategy returns")

        # Plot the portfolio value
        self.visualizer.plot_portfolio_value(self.portfolio, "Portfolio value")

        return {"Total return": total_return, "Sharpe ratio": sharpe_ratio}
 

    def get_performance_new(self):
        # Calculate the total return of the portfolio
        portfolio_return = (self.capital / self.initial_capital) - 1

        # Calculate the annualized return
        num_trades = len(self.portfolio)
        trade_duration = (self.portfolio[-1]["Exit date"] - self.portfolio[0]["Entry date"]).days
        annualized_return = (1 + portfolio_return) ** (365 / trade_duration) - 1

        # Calculate the maximum drawdown
        max_drawdown = 0
        peak = self.initial_capital
        for trade in self.portfolio:
            if trade["Exit value"] > peak:
                peak = trade["Exit value"]
            drawdown = (peak - trade["Exit value"]) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Calculate the Sharpe ratio
        returns = []
        for i in range(1, len(self.portfolio)):
            returns.append(self.portfolio[i]["Exit value"] / self.portfolio[i-1]["Entry value"] - 1)
        Sharpe_ratio = np.mean(returns) / np.std(returns)

        # Return the performance metrics
        return {
            "Total return": portfolio_return,
            "Annualized return": annualized_return,
            "Max drawdown": max_drawdown,
            "Num trades": num_trades,
            "Sharpe ratio": Sharpe_ratio
        }


class MovingAverageStrategy:
    def __init__(self, short_window, long_window):
        self.short_window = short_window
        self.long_window = long_window

    def get_risk_percentage(self, prices):
        """"""
        if len(prices) < self.long_window:
            return 0

        # Calculate the risk based on the difference between the two moving averages
        short_ma = np.mean(prices["close"][-self.short_window:])
        long_ma = np.mean(prices["close"][-self.long_window:])
  
        risk = (short_ma - long_ma) / long_ma
        return risk

    def generate_trade_signal(self, prices):
        # Calculate the moving averages
        short_ma = prices["close"].rolling(self.short_window).mean()
        long_ma = prices["close"].rolling(self.long_window).mean()

        if len(prices) < self.long_window:
            return "HOLD"

        # Check if the short MA has crossed above the long MA
        if short_ma[-2] < long_ma[-2] and short_ma[-1] > long_ma[-1]:
            return "BUY"
        # Check if the short MA has crossed below the long MA
        elif short_ma[-2] > long_ma[-2] and short_ma[-1] < long_ma[-1]:
            return "SELL"
        else:
            return "HOLD"


class RiskManager:
  def __init__(self, max_risk, stop_loss_pct):
    self.max_risk = max_risk
    self.stop_loss_pct = stop_loss_pct
    
  def get_position_size(self, prices, risk):
    # Calculate the value of the trade
    trade_value = self.max_risk * risk
    
    # Calculate the number of shares to buy
    shares = trade_value / prices["close"]
    breakpoint()
    
    return shares
  
  def get_stop_loss_price(self, buy_price):
    """unused"""
    return buy_price * (1 - self.stop_loss_pct)


class Visualizer:
    def plot_data(self, data, title):
        plt.plot(data["close"])
        plt.title(title)
        plt.xlabel("date")
        plt.ylabel("Return")
        plt.show()

    def plot_returns(self, returns, title):
        plt.plot(returns)
        plt.title(title)
        plt.xlabel("date")
        plt.ylabel("Return")
        plt.show()

    def plot_portfolio_value(self, portfolio, title):
        import pandas as pd
        plt.plot(pd.DataFrame(portfolio).set_index("date"))
        plt.title(title)
        plt.xlabel("date")
        plt.ylabel("Portfolio value")
        plt.show()


if __name__ == "__main__":
    print("test".center(88, "="))
    import yfinance as yf

    # Load the data
    # data = pd.read_csv("data.csv", index_col="date", parse_dates=True)
    data = yf.download("MSFT", "2020-01-01", "2022-01-01")
    data.columns = [x.lower() for x in data.columns]
    data.index.name = 'date'
    
    risk_manager = RiskManager(max_risk=1e3, stop_loss_pct=0.5)
    visualizer = Visualizer()
    
    backtest = Backtest(
        data, MovingAverageStrategy(20, 50), 10000, risk_manager, visualizer
    )
    
    # visualizer.plot_data(backtest.data, "underlying")
    
    # Run the backtest
    backtest.run()
    
    # Print the performance
    print(backtest.get_performance())
