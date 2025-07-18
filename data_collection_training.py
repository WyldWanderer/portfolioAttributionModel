import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import List, Dict
import pickle
import os
import torch
from typing import List, Dict

class FinancialDataCollector:
    """Collect real financial data for portfolio attribution training"""
    
    def __init__(self):
        # Full S&P 500 tickers (sample of major ones - you'd want the complete list)
        self.sp500_tickers = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'ADBE', 'CRM',
            'ORCL', 'INTC', 'AMD', 'QCOM', 'IBM', 'TXN', 'INTU', 'NOW', 'MU', 'AMAT',
            
            # Financials  
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'CME',
            'ICE', 'CB', 'PGR', 'AON', 'MMC', 'TRV', 'AIG', 'MET', 'PRU', 'AFL',
            
            # Healthcare
            'JNJ', 'PFE', 'ABT', 'MRK', 'TMO', 'DHR', 'BMY', 'ABBV', 'LLY', 'UNH',
            'CVS', 'MDT', 'GILD', 'AMGN', 'ISRG', 'SYK', 'BSX', 'EW', 'ZTS', 'REGN',
            
            # Consumer Discretionary
            'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'BKNG', 'CMG',
            'MAR', 'GM', 'F', 'CCL', 'RCL', 'NCLH', 'YUM', 'EBAY', 'ETSY', 'ABNB',
            
            # Communication Services
            'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'PARA',
            'WBD', 'EA', 'TTWO', 'MTCH', 'PINS', 'SNAP', 'TWTR', 'DISH', 'SIRI', 'NYT',
            
            # Industrials
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'GD',
            'DE', 'EMR', 'ETN', 'ITW', 'PH', 'CMI', 'FDX', 'WM', 'RSG', 'URI',
            
            # Consumer Staples
            'PG', 'KO', 'PEP', 'WMT', 'COST', 'MDLZ', 'CL', 'KMB', 'GIS', 'K',
            'HSY', 'SJM', 'CAG', 'CPB', 'MKC', 'CLX', 'TSN', 'HRL', 'CHD', 'EL',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'BKR',
            'HAL', 'DVN', 'FANG', 'APA', 'EQT', 'CTRA', 'MRO', 'HES', 'KMI', 'OKE',
            
            # Utilities
            'NEE', 'SO', 'DUK', 'AEP', 'EXC', 'XEL', 'ED', 'ETR', 'WEC', 'PPL',
            'PEG', 'ES', 'FE', 'EIX', 'DTE', 'NI', 'AES', 'LNT', 'CMS', 'CNP',
            
            # Materials
            'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'IFF',
            'ALB', 'CE', 'VMC', 'MLM', 'NUE', 'STLD', 'X', 'CF', 'MOS', 'FMC',
            
            # Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX', 'WELL', 'DLR', 'PSA', 'O', 'CBRE', 'AVB',
            'EQR', 'SBAC', 'WY', 'ARE', 'VTR', 'ESS', 'MAA', 'UDR', 'CPT', 'REG'
        ]
        
        # GICS Sector mapping for better attribution
        self.sector_mapping = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'NVDA': 'Technology',
            'META': 'Communication Services', 'NFLX': 'Communication Services', 'ADBE': 'Technology',
            
            # Financials
            'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
            
            # Healthcare  
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'ABT': 'Healthcare', 'MRK': 'Healthcare',
            
            # Add more mappings as needed...
        }
    
    def get_stock_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download stock price data for larger universe"""
        print(f"Downloading data for {len(tickers)} tickers...")
        
        stock_data = {}
        failed_tickers = []
        batch_size = 50  # Download in batches to avoid API limits
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")
            
            for ticker in batch_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date, end=end_date)
                    
                    if not hist.empty and 'Close' in hist.columns:
                        close_prices = hist['Close']
                        if isinstance(close_prices, pd.Series):
                            stock_data[ticker] = close_prices
                        else:
                            stock_data[ticker] = close_prices.squeeze()
                        
                        if len(stock_data) % 25 == 0:
                            print(f"✓ Downloaded {len(stock_data)} tickers...")
                    else:
                        failed_tickers.append(ticker)
                        
                except Exception as e:
                    failed_tickers.append(ticker)
                    if len(failed_tickers) % 10 == 0:
                        print(f"Failed tickers so far: {len(failed_tickers)}")
        
        print(f"Successfully downloaded: {len(stock_data)} tickers")
        print(f"Failed downloads: {len(failed_tickers)} tickers")
        
        if len(stock_data) < 50:
            print("Warning: Very few tickers downloaded. Using synthetic data...")
            return self.create_synthetic_stock_data(start_date, end_date)
        
        try:
            df = pd.DataFrame(stock_data)
            print(f"Stock data DataFrame shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            return self.create_synthetic_stock_data(start_date, end_date)
    
    def create_synthetic_stock_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Create synthetic stock data if real data fails"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        synthetic_data = {}
        
        for ticker in self.sp500_tickers[:200]:  # Use subset for synthetic
            # Create realistic price series
            initial_price = np.random.uniform(50, 500)
            returns = np.random.normal(0.0008, 0.02, len(dates))  # ~20% annual vol
            prices = [initial_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            synthetic_data[ticker] = pd.Series(prices, index=dates)
        
        return pd.DataFrame(synthetic_data)
    
    def generate_realistic_portfolios(self, returns_data: pd.DataFrame, n_portfolios: int = 500) -> List[Dict]:
        """Generate more realistic portfolio allocations"""
        portfolios = []
        available_assets = returns_data.columns.tolist()
        
        # Define realistic portfolio strategies
        strategy_types = [
            'large_cap_growth',    # 40-60 stocks, tech heavy
            'large_cap_value',     # 50-80 stocks, financials/industrials heavy  
            'diversified',         # 100-200 stocks, market-like weights
            'sector_focused',      # 20-40 stocks, concentrated in 1-2 sectors
            'dividend_focused'     # 50-100 stocks, utilities/consumer staples heavy
        ]
        
        for _ in range(n_portfolios):
            strategy = np.random.choice(strategy_types)
            
            if strategy == 'large_cap_growth':
                # Focus on top 50 market cap stocks with growth tilt
                n_assets = np.random.randint(40, 61)
                selected_assets = available_assets[:50]  # Assume first 50 are largest
                selected_assets = np.random.choice(selected_assets, 
                                                 min(n_assets, len(selected_assets)), 
                                                 replace=False)
                
            elif strategy == 'diversified':
                # Broader diversification
                n_assets = np.random.randint(100, 201)
                selected_assets = np.random.choice(available_assets,
                                                 min(n_assets, len(available_assets)),
                                                 replace=False)
                
            else:
                # Other strategies
                n_assets = np.random.randint(30, 81)
                selected_assets = np.random.choice(available_assets,
                                                 min(n_assets, len(available_assets)),
                                                 replace=False)
            
            # Generate more realistic weights (less extreme concentration)
            if strategy == 'diversified':
                # More equal weighting for diversified portfolios
                base_weights = np.random.dirichlet(np.ones(len(selected_assets)) * 2)
            else:
                # Some concentration for other strategies
                base_weights = np.random.dirichlet(np.ones(len(selected_assets)))
            
            # Cap maximum weight at 10% for realism
            base_weights = np.minimum(base_weights, 0.10)
            base_weights = base_weights / base_weights.sum()  # Renormalize
            
            portfolios.append({
                'assets': selected_assets.tolist(),
                'weights': base_weights.tolist(),
                'strategy': strategy,
                'rebalance_freq': np.random.choice(['M', 'Q'])
            })
        
        return portfolios
        
    def get_stock_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download stock price data"""
        print(f"Downloading data for {len(tickers)} tickers...")
        
        stock_data = {}
        failed_tickers = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if not hist.empty and 'Close' in hist.columns:
                    # Ensure we get a proper Series
                    close_prices = hist['Close']
                    if isinstance(close_prices, pd.Series):
                        stock_data[ticker] = close_prices
                    else:
                        # Handle potential DataFrame issues
                        stock_data[ticker] = close_prices.squeeze() if hasattr(close_prices, 'squeeze') else close_prices
                    print(f"✓ {ticker} - {len(close_prices)} data points")
                else:
                    print(f"✗ {ticker} - No data")
                    failed_tickers.append(ticker)
                    
            except Exception as e:
                print(f"✗ {ticker} - Error: {e}")
                failed_tickers.append(ticker)
        
        if failed_tickers:
            print(f"Failed to download {len(failed_tickers)} tickers: {failed_tickers[:5]}...")
        
        if not stock_data:
            print("Warning: No stock data downloaded. Creating dummy data...")
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            stock_data = {ticker: pd.Series(np.random.normal(100, 10, len(dates)), index=dates) 
                         for ticker in tickers[:10]}  # Just use first 10 tickers
        
        try:
            df = pd.DataFrame(stock_data)
            print(f"Stock data DataFrame shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error creating stock DataFrame: {e}")
            # Return a simple DataFrame with dummy data
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            return pd.DataFrame({
                ticker: np.random.normal(100, 10, len(dates)) 
                for ticker in tickers[:10]
            }, index=dates)
    
    def get_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get market indices and economic data"""
        market_tickers = {
            'SPY': 'S&P 500',
            '^VIX': 'VIX',
            'TLT': '20+ Year Treasury',
            'GLD': 'Gold'
        }
        
        market_data = {}
        for ticker, name in market_tickers.items():
            try:
                print(f"Downloading {name}...")
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if not hist.empty and 'Close' in hist.columns:
                    # Ensure we get a proper Series, not DataFrame
                    close_prices = hist['Close']
                    if isinstance(close_prices, pd.Series):
                        market_data[ticker.replace('^', '')] = close_prices
                    else:
                        # If it's somehow a DataFrame, take the first column
                        market_data[ticker.replace('^', '')] = close_prices.iloc[:, 0] if close_prices.ndim > 1 else close_prices
                    print(f"✓ {name} - {len(close_prices)} data points")
                else:
                    print(f"✗ {name} - No data available")
                    
            except Exception as e:
                print(f"✗ {name} - Error: {e}")
        
        if not market_data:
            print("Warning: No market data downloaded. Creating dummy data...")
            # Create dummy data if nothing downloaded
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            market_data = {
                'SPY': pd.Series(np.random.normal(400, 20, len(dates)), index=dates),
                'VIX': pd.Series(np.random.normal(20, 5, len(dates)), index=dates),
                'TLT': pd.Series(np.random.normal(100, 10, len(dates)), index=dates)
            }
        
        # Create DataFrame ensuring all series have the same index
        try:
            df = pd.DataFrame(market_data)
            print(f"Market data DataFrame shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error creating market DataFrame: {e}")
            # Fallback: align all series to common dates
            if market_data:
                common_dates = None
                for series in market_data.values():
                    if common_dates is None:
                        common_dates = series.index
                    else:
                        common_dates = common_dates.intersection(series.index)
                
                aligned_data = {}
                for ticker, series in market_data.items():
                    aligned_data[ticker] = series.reindex(common_dates)
                
                return pd.DataFrame(aligned_data)
            else:
                # Ultimate fallback
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                return pd.DataFrame({
                    'SPY': np.random.normal(400, 20, len(dates)),
                    'VIX': np.random.normal(20, 5, len(dates))
                }, index=dates)
    
    def calculate_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns"""
        return price_data.pct_change().dropna()
    
    def generate_random_portfolios(self, returns_data: pd.DataFrame, n_portfolios: int = 1000) -> List[Dict]:
        """Generate random portfolio allocations"""
        portfolios = []
        available_assets = returns_data.columns.tolist()
        
        for _ in range(n_portfolios):
            # Random number of assets (between 5 and 20)
            n_assets = np.random.randint(5, min(21, len(available_assets)))
            selected_assets = np.random.choice(available_assets, n_assets, replace=False)
            
            # Random weights that sum to 1
            weights = np.random.dirichlet(np.ones(n_assets))
            
            # Random rebalancing frequency (monthly, quarterly, etc.)
            rebalance_freq = np.random.choice(['M', 'Q', 'Y'])
            
            portfolios.append({
                'assets': selected_assets.tolist(),
                'weights': weights.tolist(),
                'rebalance_freq': rebalance_freq
            })
        
        return portfolios
    
    def calculate_portfolio_attribution(self, portfolio: Dict, returns_data: pd.DataFrame, 
                                      benchmark_returns: pd.Series) -> pd.DataFrame:
        """Calculate portfolio attribution over time"""
        assets = portfolio['assets']
        weights = np.array(portfolio['weights'])
        
        # Get returns for portfolio assets that actually exist in the data
        available_assets = [asset for asset in assets if asset in returns_data.columns]
        if not available_assets:
            print(f"Warning: No assets from portfolio found in returns data")
            return pd.DataFrame()
        
        # Adjust weights for available assets
        available_indices = [i for i, asset in enumerate(assets) if asset in available_assets]
        available_weights = weights[available_indices]
        available_weights = available_weights / available_weights.sum()  # Renormalize
        
        # Get returns for available portfolio assets
        portfolio_returns = returns_data[available_assets]
        
        # Calculate portfolio return for each date
        portfolio_return_series = (portfolio_returns * available_weights).sum(axis=1)
        
        # Find common dates between portfolio and benchmark
        common_dates = portfolio_return_series.index.intersection(benchmark_returns.index)
        
        if len(common_dates) == 0:
            print("Warning: No common dates between portfolio and benchmark")
            return pd.DataFrame()
        
        # Calculate attribution components for each date
        attribution_data = []
        
        for date in common_dates:
            port_ret = portfolio_return_series.loc[date]
            bench_ret = benchmark_returns.loc[date]
            
            # Skip if either return is NaN
            if pd.isna(port_ret) or pd.isna(bench_ret):
                continue
                
            excess_return = port_ret - bench_ret
            
            # Enhanced attribution decomposition - scale based on portfolio complexity
            excess_return = port_ret - bench_ret
            n_assets = len(available_assets)
            
            # Scale factors based on portfolio size and diversification
            diversification_factor = min(n_assets / 100, 2.0)  # More diversified = different attribution patterns
            concentration_factor = 1.0 / np.sqrt(n_assets)     # Larger portfolios = less concentration effect
            
            # Asset selection: decreases with diversification (harder to add value with more stocks)
            individual_returns = portfolio_returns.loc[date]
            weight_concentration = np.sum(available_weights ** 2)  # Herfindahl index
            
            if len(individual_returns) > 0:
                top_performer_weight = available_weights[np.argmax(individual_returns)]
                avg_weight = 1.0 / len(available_weights)
                
                # Asset selection scales DOWN with more securities (realistic!)
                base_asset_selection = (top_performer_weight - avg_weight) * excess_return
                asset_selection = base_asset_selection * concentration_factor * 0.8
            else:
                asset_selection = 0
            
            # Allocation effect: based on deviation from benchmark-like weights
            equal_weights = np.ones(len(available_weights)) / len(available_weights)
            weight_deviations = available_weights - equal_weights
            max_deviation = np.max(np.abs(weight_deviations))
            
            # Allocation effect scales with portfolio complexity
            if n_assets < 50:
                allocation_scale = 0.6  # Concentrated portfolios have bigger allocation effects
            elif n_assets < 150:
                allocation_scale = 0.4  # Medium diversification
            else:
                allocation_scale = 0.2  # Highly diversified portfolios have smaller allocation effects
            
            allocation = max_deviation * excess_return * allocation_scale
            
            # Timing effect: scales UP with diversification (more opportunities for timing)
            portfolio_vol = individual_returns.std() if len(individual_returns) > 1 else 0.02
            timing_scale = 0.1 + (diversification_factor - 1) * 0.15  # 0.1 to 0.25 range
            timing = (portfolio_vol - 0.02) * excess_return * timing_scale
            
            # Currency effect: scales UP with portfolio size (more international exposure)
            if n_assets < 30:
                currency_scale = 0.05   # Small portfolios = minimal currency exposure
            elif n_assets < 100:
                currency_scale = 0.10   # Medium portfolios = some international
            else:
                currency_scale = 0.15   # Large portfolios = significant international exposure
            
            currency = np.random.normal(0, currency_scale * abs(excess_return))
            
            # Interaction: more complex with larger portfolios
            interaction_base = 0.05 * (asset_selection * allocation) if abs(allocation) > 0 else 0
            interaction_noise = np.random.normal(0, 0.00005 * diversification_factor)
            interaction = interaction_base + interaction_noise
            
            # Ensure components sum approximately to excess return
            total_explained = asset_selection + allocation + timing + currency + interaction
            residual = excess_return - total_explained
            
            # Distribute residual proportionally (allow some unexplained variance)
            explanation_ratio = 0.7 + 0.2 * min(diversification_factor, 1.5)  # 70-100% explained
            if abs(total_explained) > 1e-6:
                scale_factor = explanation_ratio * (excess_return / total_explained) if total_explained != 0 else 1
                asset_selection *= scale_factor
                allocation *= scale_factor  
                timing *= scale_factor
                currency *= scale_factor
                
                # Interaction picks up the residual
                interaction = excess_return - (asset_selection + allocation + timing + currency)
            else:
                # If total is near zero, distribute excess return randomly
                components = np.random.dirichlet([1, 1, 1, 1, 1]) * excess_return * explanation_ratio
                asset_selection, allocation, timing, currency, interaction = components
            
            attribution_data.append({
                'date': date,  # This is a pandas Timestamp
                'portfolio_return': float(port_ret),
                'benchmark_return': float(bench_ret),
                'excess_return': float(excess_return),
                'asset_selection': float(asset_selection),
                'allocation': float(allocation),
                'timing': float(timing),
                'currency': float(currency),
                'interaction': float(interaction),
                'n_assets_used': len(available_assets),
                'portfolio_volatility': float(portfolio_returns.loc[date].std())
            })
        
        attribution_df = pd.DataFrame(attribution_data)
        
        # Ensure date column is properly formatted
        if not attribution_df.empty:
            attribution_df['date'] = pd.to_datetime(attribution_df['date'])
            
        print(f"Created attribution data for {len(attribution_df)} dates")
        return attribution_df

def create_training_dataset():
    """Create a comprehensive training dataset"""
    collector = FinancialDataCollector()
    
    # Define date range (2 years of data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Creating dataset from {start_str} to {end_str}")
    
    # Collect data
    print("\n=== Collecting stock data ===")
    stock_data = collector.get_stock_data(collector.sp500_tickers[:20], start_str, end_str)  # Use fewer tickers for testing
    
    print("\n=== Collecting market data ===")
    market_data = collector.get_market_data(start_str, end_str)
    
    # Calculate returns
    print("\n=== Calculating returns ===")
    stock_returns = collector.calculate_returns(stock_data)
    market_returns = collector.calculate_returns(market_data)
    
    print(f"Stock returns shape: {stock_returns.shape}")
    print(f"Market returns shape: {market_returns.shape}")
    
    # Use SPY as benchmark, fallback to first column if not available
    if 'SPY' in market_returns.columns:
        benchmark_returns = market_returns['SPY']
        print("Using SPY as benchmark")
    else:
        benchmark_returns = market_returns.iloc[:, 0]
        print(f"Using {market_returns.columns[0]} as benchmark")
    
    # Generate realistic portfolios
    print("\n=== Generating realistic portfolios ===")
    portfolios = collector.generate_realistic_portfolios(stock_returns, n_portfolios=200)
    
    # Calculate attribution for each portfolio
    print("\n=== Calculating attribution ===")
    all_attribution_data = []
    
    for i, portfolio in enumerate(portfolios):
        try:
            attribution_df = collector.calculate_portfolio_attribution(
                portfolio, stock_returns, benchmark_returns
            )
            
            if not attribution_df.empty:
                # Add portfolio metadata
                attribution_df['portfolio_id'] = i
                attribution_df['n_assets'] = len(portfolio['assets'])
                attribution_df['max_weight'] = max(portfolio['weights'])
                attribution_df['weight_concentration'] = sum(w**2 for w in portfolio['weights'])
                
                all_attribution_data.append(attribution_df)
            
            if i % 25 == 0:
                print(f"Processed {i}/{len(portfolios)} portfolios")
                
        except Exception as e:
            print(f"Error processing portfolio {i}: {e}")
    
    if not all_attribution_data:
        raise ValueError("No attribution data was generated. Check your data sources.")
    
    # Combine all attribution data
    print("\n=== Combining data ===")
    final_dataset = pd.concat(all_attribution_data, ignore_index=True)
    print(f"Combined dataset shape: {final_dataset.shape}")
    
    # Add market context features more carefully
    print("\n=== Adding market context ===")
    
    # Check what columns exist in final_dataset
    print(f"Final dataset columns before merge: {final_dataset.columns.tolist()}")
    if 'date' in final_dataset.columns:
        print(f"Final dataset date range: {final_dataset['date'].min()} to {final_dataset['date'].max()}")
        print(f"Final dataset date type: {type(final_dataset['date'].iloc[0])}")
    else:
        print("ERROR: No 'date' column in final_dataset!")
        return final_dataset
    
    # Prepare market returns for merge
    market_returns_with_date = market_returns.reset_index()
    
    # The index should be dates, so the reset_index creates a column from the index
    date_column_name = market_returns_with_date.columns[0]  # First column after reset_index
    market_returns_with_date.rename(columns={date_column_name: 'date'}, inplace=True)
    
    print(f"Market returns columns after reset: {market_returns_with_date.columns.tolist()}")
    if 'date' in market_returns_with_date.columns:
        print(f"Market data date range: {market_returns_with_date['date'].min()} to {market_returns_with_date['date'].max()}")
        print(f"Market data date type: {type(market_returns_with_date['date'].iloc[0])}")
    
    # Standardize date formats for merging
    try:
        # Convert both to datetime first, then to date for consistent merging
        final_dataset['date'] = pd.to_datetime(final_dataset['date']).dt.date
        market_returns_with_date['date'] = pd.to_datetime(market_returns_with_date['date']).dt.date
        
        print("Date columns standardized to datetime.date format")
        print(f"Sample final_dataset dates: {final_dataset['date'].head().tolist()}")
        print(f"Sample market data dates: {market_returns_with_date['date'].head().tolist()}")
        
        # Check for overlapping dates
        final_dates = set(final_dataset['date'])
        market_dates = set(market_returns_with_date['date'])
        overlap = final_dates.intersection(market_dates)
        print(f"Overlapping dates: {len(overlap)} out of {len(final_dates)} portfolio dates")
        
        if len(overlap) > 0:
            # Perform the merge
            print("Performing merge...")
            final_dataset = final_dataset.merge(
                market_returns_with_date, 
                on='date',
                how='left',
                suffixes=('', '_market')
            )
            print(f"Final dataset shape after merge: {final_dataset.shape}")
        else:
            print("Warning: No overlapping dates found. Skipping market data merge.")
            
    except Exception as merge_error:
        print(f"Error during date processing or merge: {merge_error}")
        print("Continuing without market data merge...")
    
    # Fill any remaining NaN values
    final_dataset = final_dataset.fillna(0)
    
    # Save dataset
    print("\n=== Saving dataset ===")
    final_dataset.to_csv('portfolio_attribution_dataset.csv', index=False)
    
    # Save additional metadata
    with open('portfolio_metadata.pkl', 'wb') as f:
        pickle.dump({
            'portfolios': portfolios,
            'stock_tickers': collector.sp500_tickers[:20],
            'date_range': (start_str, end_str)
        }, f)
    
    print(f"Dataset created with {len(final_dataset)} samples")
    print(f"Final columns: {final_dataset.columns.tolist()}")
    return final_dataset

def train_model_with_real_data():
    """Train the model with real financial data"""
    from portfolio_attribution_model import PortfolioAttributionEngine
    
    # Load or create dataset
    if os.path.exists('portfolio_attribution_dataset.csv'):
        print("Loading existing dataset...")
        dataset = pd.read_csv('portfolio_attribution_dataset.csv')
    else:
        print("Creating new dataset...")
        dataset = create_training_dataset()
    
    print(f"Dataset shape: {dataset.shape}")
    print(f"Dataset columns: {dataset.columns.tolist()}")
    
    # Clean the dataset
    dataset = dataset.dropna(subset=['asset_selection', 'allocation', 'timing', 'currency', 'interaction'])
    
    # Prepare features and targets
    feature_columns = [
        'portfolio_return', 'benchmark_return', 'excess_return',
        'n_assets', 'max_weight', 'weight_concentration'
    ]
    
    # Add market data columns if available
    market_columns = ['SPY', 'VIX', 'TLT', 'GLD']
    for col in market_columns:
        if col in dataset.columns:
            feature_columns.append(col)
            print(f"Added market feature: {col}")
    
    target_columns = ['asset_selection', 'allocation', 'timing', 'currency', 'interaction']
    
    # Check if all required columns exist
    missing_features = [col for col in feature_columns if col not in dataset.columns]
    missing_targets = [col for col in target_columns if col not in dataset.columns]
    
    if missing_features:
        print(f"Warning: Missing feature columns: {missing_features}")
        feature_columns = [col for col in feature_columns if col in dataset.columns]
    
    if missing_targets:
        print(f"Error: Missing target columns: {missing_targets}")
        return None
    
    # Create feature matrix
    features = dataset[feature_columns].fillna(0)
    targets = dataset[target_columns].fillna(0)
    
    print(f"Training with {len(features)} samples and {len(feature_columns)} features")
    print(f"Feature columns: {feature_columns}")
    
    # Initialize and train model with much more conservative settings
    attribution_engine = PortfolioAttributionEngine({
        'hidden_dims': [32, 16],       # Much simpler architecture
        'dropout_rate': 0.5,           # Heavy regularization
        'learning_rate': 0.001,        # Much lower learning rate
        'batch_size': 64,              # Larger batches for stability
        'epochs': 150
    })
    
    # Override the feature columns since we're using real data
    attribution_engine.feature_columns = feature_columns
    
    # Prepare data
    train_loader, test_loader = attribution_engine.prepare_data(features, targets)
    
    # Train model
    print("Training model...")
    train_losses, test_losses = attribution_engine.train_model(train_loader, test_loader)
    
    # Evaluate model
    metrics = attribution_engine.evaluate_model(test_loader)
    
    print("\nFinal Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # Save trained model with robust error handling
    print("\n💾 Saving trained model...")
    
    try:
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': attribution_engine.model.state_dict(),
            'model_params': attribution_engine.model_params,
            'scaler_features': attribution_engine.scaler_features,
            'scaler_targets': attribution_engine.scaler_targets,
            'feature_columns': attribution_engine.feature_columns,
            'target_columns': attribution_engine.target_columns,
            'input_dim': len(attribution_engine.feature_columns),
            'training_metrics': metrics,  # Include performance metrics
            'timestamp': datetime.now().isoformat()
        }
        
        # Save with multiple backups for safety
        temp_file = 'trained_attribution_model_temp.pth'
        final_file = 'trained_attribution_model.pth'
        backup_file = 'trained_attribution_model_backup.pth'
        
        # Step 1: Save to temporary file
        torch.save(checkpoint_data, temp_file)
        print(f"✅ Saved to temporary file: {temp_file}")
        
        # Step 2: Verify the temporary file works
        try:
            test_load = torch.load(temp_file, map_location='cpu', weights_only=False)
            print("✅ Temporary file verified")
        except Exception as verify_error:
            raise Exception(f"Temporary file verification failed: {verify_error}")
        
        # Step 3: Create backup of existing file if it exists
        if os.path.exists(final_file):
            if os.path.exists(backup_file):
                os.remove(backup_file)
            os.rename(final_file, backup_file)
            print(f"📋 Backed up previous model to: {backup_file}")
        
        # Step 4: Move temp to final location (atomic operation)
        os.rename(temp_file, final_file)
        print(f"✅ Model saved successfully to: {final_file}")
        
        # Step 5: Verify final file
        try:
            final_test = torch.load(final_file, map_location='cpu', weights_only=False)
            print("✅ Final file verified")
            print(f"📊 Model R²: {metrics.get('portfolio_total_r2', 'Unknown'):.3f}")
            print(f"🎯 Directional Accuracy: {metrics.get('portfolio_directional_accuracy', 'Unknown'):.1f}%")
        except Exception as final_verify_error:
            print(f"❌ Final file verification failed: {final_verify_error}")
            # Restore backup if available
            if os.path.exists(backup_file):
                os.rename(backup_file, final_file)
                print("🔄 Restored from backup")
    
    except Exception as save_error:
        print(f"❌ Error saving model: {save_error}")
        print("💡 Model training completed but save failed - you may need to retrain")
        return None
    
    return attribution_engine

if __name__ == "__main__":
    # Install required packages first:
    # pip install yfinance pandas numpy scikit-learn torch matplotlib seaborn
    
    # Train the model with real data
    trained_model = train_model_with_real_data()