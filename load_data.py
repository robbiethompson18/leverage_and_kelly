import pandas as pd
import numpy as np


def load_sp500():
    """
    Load and clean S&P 500 historical data.

    Returns:
        pd.DataFrame: Cleaned S&P 500 data with columns:
            - Date: datetime
            - Price, Open, High, Low: float
            - Vol.: string (empty)
            - Change %: float (as decimal, not percentage)
            - Raw Change (bps): float (unadjusted change in basis points)
            - Days Since Prev: int (days between this row and previous)
    """
    # Load S&P 500 data with all columns as strings initially
    sp500 = pd.read_csv('data/sp500.csv', on_bad_lines='skip', encoding='utf-8-sig', dtype=str)

    # Convert Date column to datetime
    sp500['Date'] = pd.to_datetime(sp500['Date'])

    # Clean numeric columns (remove commas and convert to float)
    numeric_cols = ['Price', 'Open', 'High', 'Low']
    for col in numeric_cols:
        if col in sp500.columns:
            sp500[col] = sp500[col].str.replace(',', '').astype(float)

    # Clean Change % column
    if 'Change %' in sp500.columns:
        sp500['Change %'] = sp500['Change %'].str.rstrip('%').astype(float) / 100

    # Sort by date (oldest first) before calculating days since prev
    sp500 = sp500.sort_values('Date').reset_index(drop=True)

    # Add Raw Change in basis points (1% = 100 bps)
    sp500['Raw Change (bps)'] = sp500['Change %'] * 10000

    # Calculate days since previous date
    sp500['Days Since Prev'] = sp500['Date'].diff().dt.days
    sp500['Days Since Prev'] = sp500['Days Since Prev'].fillna(0).astype(int)

    return sp500


def load_fedfunds():
    """
    Load Federal Funds Rate data.

    Returns:
        pd.DataFrame: Federal Funds Rate data with columns:
            - Date: datetime
            - FedRate: float (percentage)
    """
    fedfunds = pd.read_csv('data/FEDFUNDS.csv')
    fedfunds['observation_date'] = pd.to_datetime(fedfunds['observation_date'])
    fedfunds = fedfunds.rename(columns={'observation_date': 'Date', 'FEDFUNDS': 'FedRate'})
    fedfunds = fedfunds.sort_values('Date').reset_index(drop=True)

    return fedfunds


def load_all_data():
    """
    Load both S&P 500 and Federal Funds Rate data.

    Returns:
        tuple: (sp500_df, fedfunds_df)
    """
    return load_sp500(), load_fedfunds()


def adjust_returns_for_dividends(sp500_df, annual_div_yield=0.02, annual_expense_ratio=0.0003):
    """
    Adjust S&P 500 returns to account for dividends and expense ratios.

    The raw data from Investing.com does NOT include dividends, so we need to add them.
    We also subtract the VOO expense ratio (default 3 bps = 0.0003).

    Dividends and expenses are compounded based on the actual number of days elapsed
    (e.g., 3 days for weekends), using the formula: (1 + annual)^(days/365) - 1

    Args:
        sp500_df: DataFrame with S&P 500 data (must have 'Days Since Prev' column)
        annual_div_yield: Annual dividend yield as decimal (e.g., 0.02 for 2%)
                         Can also be a dict mapping year ranges to yields, e.g.:
                         {(1975, 1982): 0.045, (1983, 1989): 0.035, ...}
        annual_expense_ratio: Annual expense ratio as decimal (default 0.0003 for 3 bps)

    Returns:
        pd.DataFrame: Copy of input with new columns:
            - Daily Div Yield (bps): dividend yield for the period
            - Expense (bps): expense ratio for the period
            - Total Return (bps): raw + dividends - expenses

    Example:
        # Constant 2% dividend yield
        adjusted = adjust_returns_for_dividends(sp500, annual_div_yield=0.02)

        # Time-varying dividend yields (based on README historical estimates)
        div_schedule = {
            (1975, 1982): 0.045,  # 4.5% mid-70s to early-80s
            (1983, 1989): 0.035,  # 3.5% late-80s
            (1990, 1999): 0.025,  # 2.5% 1990s
            (2000, 2019): 0.020,  # 2.0% 2000s-2010s
            (2020, 2025): 0.0125, # 1.25% mid-2020s
        }
        adjusted = adjust_returns_for_dividends(sp500, annual_div_yield=div_schedule)
    """
    df = sp500_df.copy()

    # Handle first row (Days Since Prev = 0) by treating it as 1 day
    days_elapsed = df['Days Since Prev'].replace(0, 1)

    if isinstance(annual_div_yield, dict):
        # Time-varying dividend yields
        df['Year'] = df['Date'].dt.year
        df['Annual Div Yield'] = 0.0

        for (start_year, end_year), div_yield in annual_div_yield.items():
            mask = (df['Year'] >= start_year) & (df['Year'] <= end_year)
            df.loc[mask, 'Annual Div Yield'] = div_yield

        # Calculate dividend for actual days elapsed: (1 + annual)^(days/365) - 1
        df['Daily Div Yield (bps)'] = (((1 + df['Annual Div Yield']) ** (days_elapsed / 365) - 1) * 10000)
        df = df.drop(['Year', 'Annual Div Yield'], axis=1)
    else:
        # Constant dividend yield, compounded for actual days elapsed
        df['Daily Div Yield (bps)'] = ((1 + annual_div_yield) ** (days_elapsed / 365) - 1) * 10000

    # Expense ratio compounded for actual days elapsed
    df['Expense (bps)'] = ((1 + annual_expense_ratio) ** (days_elapsed / 365) - 1) * 10000

    # Total return = raw return + dividend yield - expense ratio (all in bps)
    df['Total Return (bps)'] = df['Raw Change (bps)'] + df['Daily Div Yield (bps)'] - df['Expense (bps)']

    return df


def calculate_leveraged_returns(sp500_df, fedfunds_df, leverage=2.0, margin_spread=0.01):
    """
    Calculate leveraged returns accounting for borrowing costs.

    Borrowing cost = Risk-Free Rate (from FEDFUNDS) + margin spread (default 1%)
    Cost is applied to the borrowed amount: (leverage - 1) × position
    Cost compounds over multi-day periods (weekends, holidays) just like dividends.

    Args:
        sp500_df: DataFrame with S&P 500 data (must have 'Total Return (bps)' and 'Days Since Prev')
                 Should be output from adjust_returns_for_dividends()
        fedfunds_df: DataFrame with Federal Funds Rate data
        leverage: Leverage multiple (e.g., 2.0 for 2x leverage, 1.5 for 1.5x)
        margin_spread: Additional spread paid on margin (default 0.01 for 1% per year)

    Returns:
        pd.DataFrame: Copy with new columns:
            - Daily RFR (bps): risk-free rate for the period
            - Daily Leverage Cost (bps): total borrowing cost (RFR + spread) for the period
            - Leveraged Return (bps): return after accounting for leverage and borrowing costs

    Example:
        sp500, fedfunds = load_all_data()
        sp500 = adjust_returns_for_dividends(sp500, annual_div_yield=0.02)
        leveraged = calculate_leveraged_returns(sp500, fedfunds, leverage=2.0)
    """
    df = sp500_df.copy()

    # Merge FEDFUNDS data using merge_asof to match daily data with monthly rates
    # This matches each S&P date with the most recent FEDFUNDS rate
    df = pd.merge_asof(df.sort_values('Date'),
                       fedfunds_df[['Date', 'FedRate']].sort_values('Date'),
                       on='Date',
                       direction='backward')

    # Forward-fill any remaining NaNs (shouldn't be any, but just in case)
    df['FedRate'] = df['FedRate'].ffill().bfill()

    # Handle days since prev (0 -> 1 for first row)
    days_elapsed = df['Days Since Prev'].replace(0, 1)

    # Calculate daily RFR (compounded over actual days)
    # FedRate is in percentage (e.g., 4.33), convert to decimal
    df['Daily RFR (bps)'] = ((1 + df['FedRate']/100) ** (days_elapsed / 365) - 1) * 10000

    # Calculate daily leverage cost (RFR + spread, compounded over actual days)
    annual_leverage_cost = df['FedRate']/100 + margin_spread
    df['Daily Leverage Cost (bps)'] = ((1 + annual_leverage_cost) ** (days_elapsed / 365) - 1) * 10000

    # Calculate leveraged return
    # Formula: (Total Return × leverage) - ((leverage - 1) × borrowing cost)
    borrowed_amount = leverage - 1
    df['Leveraged Return (bps)'] = (df['Total Return (bps)'] * leverage) - (borrowed_amount * df['Daily Leverage Cost (bps)'])

    return df


def calculate_kelly_metrics(returns_bps, dates=None):
    """
    Calculate cumulative return and Kelly utility from a series of returns.

    Kelly utility is the log of the cumulative return multiplier, which equals
    the sum of log(1 + return) for each period. This is what the Kelly criterion
    maximizes over time.

    Args:
        returns_bps: Series or array of returns in basis points
        dates: Optional Series of dates to calculate calendar days

    Returns:
        dict with:
            - cumulative_return: Total return multiplier (e.g., 2.5 = 2.5x your money)
            - kelly_utility: Log of cumulative return = sum(log(1 + return))
            - periods: Number of return periods
            - calendar_days: Number of calendar days (if dates provided)

    Example:
        >>> returns = pd.Series([100, -50, 200])  # 1%, -0.5%, 2% in bps
        >>> metrics = calculate_kelly_metrics(returns)
        >>> print(f"Your money multiplied by {metrics['cumulative_return']:.2f}x")
        >>> print(f"Kelly utility: {metrics['kelly_utility']:.4f}")
    """
    # Convert bps to decimal returns
    returns_decimal = returns_bps / 10000

    # Floor at -99.99% to prevent log(0) or log(negative)
    # This simulates margin call/liquidation
    returns_decimal = returns_decimal.clip(lower=-0.9999)

    # Calculate 1 + return for each period
    growth_factors = 1 + returns_decimal

    # Cumulative return = product of all growth factors
    cumulative_return = growth_factors.prod()

    # Kelly utility = log of cumulative return = sum of log(1 + return)
    # This is numerically more stable than log(product)
    kelly_utility = np.log(growth_factors).sum()

    result = {
        'cumulative_return': cumulative_return,
        'kelly_utility': kelly_utility,
        'periods': len(returns_bps)
    }

    # Calculate calendar days if dates provided
    if dates is not None:
        calendar_days = (dates.max() - dates.min()).days
        result['calendar_days'] = calendar_days

    return result


def bootstrap_returns(df, n_periods=10000, min_block=3, max_block=10, n_simulations=1, random_seed=None):
    """
    Bootstrap returns using random block sampling with replacement.

    Samples random blocks of consecutive returns to preserve short-term autocorrelation
    (like volatility clustering) while creating new synthetic return sequences.

    VECTORIZED: Pre-generates all random numbers for all simulations at once,
    then builds each bootstrap sample using those pre-generated numbers.

    The difference between return_period_number and return_period_days_elapsed:
    - return_period_number: Just a counter (0, 1, 2, 3, ...)
    - return_period_days_elapsed: Cumulative sum of 'Days Since Prev' - accounts for
      weekends/holidays in the bootstrapped sequence

    Args:
        df: DataFrame with returns (must have 'Days Since Prev' column)
        n_periods: Target number of return periods to generate (default 10000)
        min_block: Minimum block length in periods (default 3)
        max_block: Maximum block length in periods (default 10)
        n_simulations: Number of independent bootstrap sequences (default 1)
        random_seed: Optional random seed for reproducibility

    Returns:
        dict mapping simulation number (0 to n_simulations-1) to DataFrame with:
            - return_period_number: Sequential counter (0, 1, 2, ...)
            - All return columns from input df
            - return_period_days_elapsed: Cumulative calendar days

    Example:
        >>> # Single simulation
        >>> boot = bootstrap_returns(leveraged_df, n_periods=10000)
        >>> boot[0][['return_period_number', 'Leveraged Return (bps)', 'return_period_days_elapsed']].head()

        >>> # Multiple simulations for Monte Carlo
        >>> boot = bootstrap_returns(leveraged_df, n_periods=10000, n_simulations=1000)
        >>> for sim_num, sim_df in boot.items():
        >>>     metrics = calculate_kelly_metrics(sim_df['Leveraged Return (bps)'])
    """
    # Set random seed once at the start
    rng = np.random.RandomState(random_seed)

    # Pre-calculate max number of blocks we might need
    # Worst case: all blocks are min_block length
    max_blocks_needed = int(np.ceil(n_periods / min_block)) + 1

    # Pre-generate ALL random numbers for all simulations at once
    # Shape: (n_simulations, max_blocks_needed)
    block_lengths_all = rng.randint(min_block, max_block + 1, size=(n_simulations, max_blocks_needed))
    start_indices_all = rng.randint(0, len(df), size=(n_simulations, max_blocks_needed))

    results = {}

    # Now build each simulation using the pre-generated random numbers
    for sim_num in range(n_simulations):
        bootstrapped_blocks = []
        current_period = 0
        block_idx = 0

        while current_period < n_periods:
            # Use pre-generated random numbers
            block_length = block_lengths_all[sim_num, block_idx]
            start_idx = start_indices_all[sim_num, block_idx]
            block_idx += 1

            # Calculate end index (truncate if goes past end of data)
            end_idx = min(start_idx + block_length, len(df))

            # Extract block
            block = df.iloc[start_idx:end_idx].copy()

            # Add to bootstrapped sequence
            bootstrapped_blocks.append(block)
            current_period += len(block)

        # Concatenate all blocks
        bootstrapped_df = pd.concat(bootstrapped_blocks, ignore_index=True)

        # Truncate to exact n_periods
        bootstrapped_df = bootstrapped_df.iloc[:n_periods].copy()

        # Add return_period_number (simple sequential counter)
        bootstrapped_df.insert(0, 'return_period_number', range(n_periods))

        # Calculate cumulative calendar days elapsed
        if 'Days Since Prev' in bootstrapped_df.columns:
            bootstrapped_df['return_period_days_elapsed'] = bootstrapped_df['Days Since Prev'].cumsum()

        results[sim_num] = bootstrapped_df

    return results


if __name__ == "__main__":
    # Load data
    sp500, fedfunds = load_all_data()

    # Display basic info
    print("S&P 500 Data:")
    print(f"  Date range: {sp500['Date'].min()} to {sp500['Date'].max()}")
    print(f"  Total rows: {len(sp500)}")
    print(f"\n{sp500.head()}\n")

    print("\nFederal Funds Rate Data:")
    print(f"  Date range: {fedfunds['Date'].min()} to {fedfunds['Date'].max()}")
    print(f"  Total rows: {len(fedfunds)}")
    print(f"\n{fedfunds.head()}\n")

    # Calculate some basic statistics
    print("\nS&P 500 Statistics:")
    print(sp500[['Price', 'Raw Change (bps)', 'Days Since Prev']].describe())

    # Show dividend adjustment example
    print("\n\nDividend Adjustment Example:")
    print("Adjusting with constant 2% annual dividend yield...")
    adjusted = adjust_returns_for_dividends(sp500, annual_div_yield=0.02)
    print(f"\nRaw avg daily return: {sp500['Raw Change (bps)'].mean():.2f} bps")
    print(f"Total avg daily return: {adjusted['Total Return (bps)'].mean():.2f} bps")
    print(f"Dividend contribution: {adjusted['Daily Div Yield (bps)'].mean():.2f} bps")
    print(f"Expense drag: {adjusted['Expense (bps)'].mean():.2f} bps")

    # Show leveraged returns example
    print("\n\nLeveraged Returns Example (2x leverage):")
    leveraged = calculate_leveraged_returns(adjusted, fedfunds, leverage=2.0, margin_spread=0.01)
    print(f"\nUnlevered return: {adjusted['Total Return (bps)'].mean():.2f} bps/day")
    print(f"2x Leveraged return: {leveraged['Leveraged Return (bps)'].mean():.2f} bps/day")
    print(f"Avg borrowing cost: {leveraged['Daily Leverage Cost (bps)'].mean():.2f} bps/day")
    print(f"Avg RFR: {leveraged['Daily RFR (bps)'].mean():.2f} bps/day")

    # Compare Kelly utilities across leverage levels
    print("\n\nKelly Utility Comparison:")
    print("="*80)
    print(f"{'Leverage':<10} {'Cumulative':<15} {'Kelly Utility':<15} {'Periods':<10} {'Cal Days':<10}")
    print("="*80)

    leverage_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    results_list = []

    # Calculate metrics for all leverage levels
    for lev in leverage_levels:
        if lev == 1.0:
            metrics = calculate_kelly_metrics(adjusted['Total Return (bps)'], dates=adjusted['Date'])
        else:
            lev_data = calculate_leveraged_returns(adjusted, fedfunds, leverage=lev)
            metrics = calculate_kelly_metrics(lev_data['Leveraged Return (bps)'], dates=lev_data['Date'])

        results_list.append((lev, metrics))

    # Find optimal leverage
    optimal_leverage = max(results_list, key=lambda x: x[1]['kelly_utility'])[0]

    # Print results
    for lev, metrics in results_list:
        marker = " ← Optimal" if lev == optimal_leverage else ""
        print(f"{lev:<10.1f} {metrics['cumulative_return']:<15.2f} {metrics['kelly_utility']:<15.4f} "
              f"{metrics['periods']:<10} {metrics.get('calendar_days', 'N/A'):<10}{marker}")

    print("="*80)
    print("Note: Kelly utility = log(cumulative return)")
    print(f"Optimal leverage: {optimal_leverage}x maximizes long-term growth rate")

    # Print summary stats
    first_metrics = results_list[0][1]
    if 'calendar_days' in first_metrics:
        years = first_metrics['calendar_days'] / 365.25
        print(f"Time period: {first_metrics['calendar_days']} calendar days ({years:.1f} years)")

    # Bootstrap example
    print("\n\nBootstrap Example:")
    print("="*80)
    print("Generating 3 bootstrap simulations with 2x leverage...")
    lev_2x = calculate_leveraged_returns(adjusted, fedfunds, leverage=2.0)
    bootstrap_sims = bootstrap_returns(lev_2x, n_periods=10000, n_simulations=3, random_seed=42)

    print(f"\nGenerated {len(bootstrap_sims)} simulations")
    print("\nSample from simulation 0:")
    print(bootstrap_sims[0][['return_period_number', 'Leveraged Return (bps)', 'Days Since Prev', 'return_period_days_elapsed']].head(10))

    print("\nBootstrap simulation metrics:")
    for sim_num, sim_df in bootstrap_sims.items():
        metrics = calculate_kelly_metrics(sim_df['Leveraged Return (bps)'])
        print(f"  Sim {sim_num}: Cumulative return = {metrics['cumulative_return']:.2f}x, "
              f"Kelly utility = {metrics['kelly_utility']:.4f}, "
              f"Periods = {metrics['periods']}")
