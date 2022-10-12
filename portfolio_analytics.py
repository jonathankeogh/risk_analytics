import yfinance as yf
import datetime as datetime
from pprint import pprint
from dateutil import relativedelta
import numpy as np
from scipy import stats


def timeseries_returns(ts):
    '''
    Gives the absolute, not logarithmic, returns
    :param ts: (list of floats)
    :return: returns of ts (list of floats)
    '''
    return [ts[i + 1] / ts[i] - 1 for i in range(0, len(ts) - 1)]

def get_one_year_timeseries(ticker):
    '''
    Returns one year EOD prices from yahoo finance
    :param ticker: (str)
    :return: (dict - {datetime: float})
    '''

    # Initialize yf object
    stock = yf.Ticker(ticker)

    # Pull one year timeseries
    one_year_df = stock.history(period='1y')

    # Use EOD prices
    ts = one_year_df.to_dict()['Close']

    # Use datetime for timestamps
    ts = {date.to_pydatetime().replace(tzinfo=None): price for date, price in ts.items()}

    return ts


def pull_underlying_portfolio_timeseries(tickers_and_weights):
    # Make sure weights add up to 1
    assert round(float(sum(ticker_and_weight[1] for ticker_and_weight in tickers_and_weights)), 5) == 1.0

    # Loop over every ticker and pull their timeseries from yahoo finance
    aggregate_timeseries = []
    for astock in tickers_and_weights:
        # Stock data
        ticker = astock[0]
        weight = astock[1]

        # Pull one year timeseries
        ts = get_one_year_timeseries(ticker)

        # Weight the timeseries
        ts = {timestamp: price * weight for timestamp, price in ts.items()}

        aggregate_timeseries.append(ts)

    return aggregate_timeseries

def portfolio_analytics(portfolio):
    """
    This routine calculates the alpha, beta, correlation and Sharpe ratio of a portfolio with respect to the S&P
    :param portfolio: (list of tuples of (str, float)) - the tickers and weights of the whole portfolio
    """

    # Get the underlyings timeseries
    aggregate_ts = pull_underlying_portfolio_timeseries(portfolio)

    # Pull the S&P
    snp = get_one_year_timeseries('^GSPC')

    # Get the intersection of all the timeseries' timestamps
    dates = [adate for adate in aggregate_ts[0] if all(adate in ts for ts in aggregate_ts) and adate in snp]

    # Add all together to get Portfolio timeseries
    portfolio_ts = [sum(ts[adate] for ts in aggregate_ts) for adate in dates]

    # Get price differences for VaR
    price_diffs = [portfolio_ts[i + 1] - portfolio_ts[i] for i in range(0, len(dates) - 1)]
    price_diffs.sort()

    # Give the VaR
    print('99% 1month VaR is ', price_diffs[2] * np.sqrt(20))

    # Collect the returns
    snp_returns = timeseries_returns([snp[adate] for adate in dates])
    portfolio_ts_returns = timeseries_returns(portfolio_ts)

    # Run the linear regression
    (beta, alpha, corr) = stats.linregress(snp_returns, portfolio_ts_returns)[0:3]

    # Print the results
    print('Alpha is ', alpha)
    print('Beta is ', alpha)
    print('Correlation is ', corr)
    print('Sharpe Ratio is ', np.mean(portfolio_ts_returns) * 252 ** 0.5 / np.std(portfolio_ts_returns))

if __name__ == "__main__":

    pauls_portfolio = [
        ('SCHD', 0.3),
        ('NOBL', 0.2),
        ('QYLD', 0.03),
        ('O', 0.06),
        ('C', 0.05),
        ('ARCC', 0.05),
        ('RMG.L', 0.03),
        ('ENB', 0.05),
        ('MO', 0.05),
        ('HTGC', 0.03),
        ('MAIN', 0.04),
        ('AMCR', 0.03),
        ('LEG', 0.03),
        ('WPC', 0.05),
    ]

    portfolio_analytics(pauls_portfolio)
