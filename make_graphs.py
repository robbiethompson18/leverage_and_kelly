import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import importlib
import load_data
importlib.reload(load_data)

from load_data import (load_all_data, adjust_returns_for_dividends,
                       calculate_leveraged_returns, bootstrap_returns,
                       calculate_kelly_metrics)

# Set seaborn style
sns.set_style("darkgrid")
sns.set_palette("husl")


def calculate_portfolio_value(returns_bps, initial_value=1000):
    """
    Calculate cumulative portfolio value from daily returns in bps.

    Args:
        returns_bps: Series of daily returns in basis points
        initial_value: Starting portfolio value (default $1000)

    Returns:
        Series of cumulative portfolio values
    """
    # Convert bps to decimal returns: bps / 10000
    daily_returns = returns_bps / 10000

    # Floor returns at -100% (can't lose more than everything in one day)
    # This simulates margin call / liquidation
    daily_returns = daily_returns.clip(lower=-0.9999)

    # Calculate cumulative returns: (1 + r1) * (1 + r2) * ...
    cumulative_return = (1 + daily_returns).cumprod()

    # Multiply by initial value
    portfolio_value = initial_value * cumulative_return

    return portfolio_value


def plot_leverage_comparison(leverage_levels=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                             annual_div_yield=0.02,
                             margin_spread=0.01,
                             initial_value=1000,
                             log_scale=False):
    """
    Plot portfolio value over time for different leverage levels.

    Args:
        leverage_levels: List of leverage multiples to compare
        annual_div_yield: Annual dividend yield (default 2%)
        margin_spread: Margin spread over RFR (default 1%)
        initial_value: Starting portfolio value (default $1000)
        log_scale: Use logarithmic scale for y-axis (default False)
    """
    # Load data
    print("Loading data...")
    sp500, fedfunds = load_all_data()

    # Adjust for dividends
    print("Adjusting for dividends...")
    sp500_adjusted = adjust_returns_for_dividends(sp500, annual_div_yield=annual_div_yield)

    # Create figure
    plt.figure(figsize=(14, 8))

    # Calculate and plot each leverage level
    results = {}
    for leverage in leverage_levels:
        print(f"Calculating {leverage}x leverage...")

        if leverage == 1.0:
            # No leverage case - just use total return
            portfolio_value = calculate_portfolio_value(
                sp500_adjusted['Total Return (bps)'],
                initial_value
            )
            label = f"{leverage}x (No Leverage)"
        else:
            # Leveraged case
            leveraged = calculate_leveraged_returns(
                sp500_adjusted,
                fedfunds,
                leverage=leverage,
                margin_spread=margin_spread
            )
            portfolio_value = calculate_portfolio_value(
                leveraged['Leveraged Return (bps)'],
                initial_value
            )
            label = f"{leverage}x Leverage"

        # Store results
        results[leverage] = portfolio_value

        # Plot
        plt.plot(sp500_adjusted['Date'], portfolio_value, label=label, linewidth=2)

    # Format plot
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(f'Portfolio Value ($)' + (' (Log Scale)' if log_scale else ''), fontsize=12)
    plt.title(f'Growth of ${initial_value:,} at Different Leverage Levels\n' +
              f'(Div Yield: {annual_div_yield*100:.1f}%, Margin Spread: {margin_spread*100:.1f}%)',
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)

    # Set log scale if requested
    if log_scale:
        plt.yscale('log')

    plt.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Print final values
    print("\n" + "="*60)
    print(f"Final Portfolio Values (starting with ${initial_value:,}):")
    print("="*60)
    for leverage in leverage_levels:
        final_value = results[leverage].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100
        print(f"{leverage:4.1f}x: ${final_value:>15,.2f}  ({total_return:>8.1f}% total return)")
    print("="*60)

    return results


def run_bootstrap_simulations(sp500_adjusted, fedfunds, leverage_levels, n_simulations=10,
                              n_periods=10000, min_block=3, max_block=10, random_seed=None):
    """
    Run bootstrap simulations across multiple leverage levels.

    VECTORIZED APPROACH: Generates bootstrap samples ONCE, then applies all leverage
    levels to each bootstrap sample. This ensures:
    1. Each simulation is truly independent (different bootstrap sample)
    2. Faster execution (only bootstrap once instead of once per leverage level)
    3. Fixes the correlation issue (no more linear bands in scatter plots)

    Args:
        sp500_adjusted: DataFrame with dividend-adjusted S&P 500 data
        fedfunds: DataFrame with Federal Funds Rate data
        leverage_levels: List of leverage levels to test (e.g., [1.0, 1.5, 2.0, 2.5, 3.0])
        n_simulations: Number of independent bootstrap simulations (default 10)
        n_periods: Number of periods per simulation (default 10000)
        min_block: Minimum block length for bootstrap (default 3)
        max_block: Maximum block length for bootstrap (default 10)
        random_seed: Optional random seed for reproducibility

    Returns:
        list of dicts, each containing:
            - leverage: The leverage level
            - simulation: Simulation number (0 to n_simulations-1)
            - cumulative_return: Total return multiplier
            - kelly_utility: Kelly utility value
            - periods: Number of periods
    """
    results = []

    # Generate bootstrap samples ONCE from unleveraged data
    print(f"Generating {n_simulations} bootstrap samples...")
    bootstrap_samples = bootstrap_returns(sp500_adjusted, n_periods=n_periods,
                                         min_block=min_block, max_block=max_block,
                                         n_simulations=n_simulations,
                                         random_seed=random_seed)

    # Now apply each leverage level to each bootstrap sample
    print(f"Applying {len(leverage_levels)} leverage levels to each bootstrap...")
    for sim_num, boot_df in bootstrap_samples.items():
        if sim_num % 5 == 0:
            print(f"  Processing simulation {sim_num+1}/{n_simulations}...")

        for leverage in leverage_levels:
            # Apply leverage to this bootstrap sample
            if leverage == 1.0:
                # No leverage - use total returns directly
                returns_bps = boot_df['Total Return (bps)']
            else:
                # Apply leverage calculation to bootstrapped data
                # Need to recalculate leveraged returns for this bootstrap sample
                lev_boot = calculate_leveraged_returns(boot_df, fedfunds, leverage=leverage)
                returns_bps = lev_boot['Leveraged Return (bps)']

            # Calculate Kelly metrics
            metrics = calculate_kelly_metrics(returns_bps)
            results.append({
                'leverage': leverage,
                'simulation': sim_num,
                'cumulative_return': metrics['cumulative_return'],
                'kelly_utility': metrics['kelly_utility'],
                'periods': metrics['periods']
            })

    print(f"Completed {len(results)} total evaluations ({n_simulations} sims × {len(leverage_levels)} leverage levels)")
    return results


def plot_bootstrap_scatter(bootstrap_results, title="Bootstrap Simulation Results"):
    """
    Create scatter plot of bootstrap simulation results.

    Args:
        bootstrap_results: List of dicts from run_bootstrap_simulations()
        title: Plot title

    Returns:
        matplotlib figure
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(bootstrap_results)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Create scatter plot with color by leverage level
    leverage_levels = sorted(df['leverage'].unique())
    colors = sns.color_palette("husl", len(leverage_levels))

    for i, lev in enumerate(leverage_levels):
        lev_data = df[df['leverage'] == lev]
        plt.scatter(lev_data['cumulative_return'],
                   lev_data['kelly_utility'],
                   c=[colors[i]],
                   label=f'{lev}x Leverage',
                   s=100,
                   alpha=0.7,
                   edgecolors='black',
                   linewidth=1)

    plt.xlabel('Cumulative Return (multiplier)', fontsize=12)
    plt.ylabel('Kelly Utility', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_kelly_vs_leverage(bootstrap_results, title="Kelly Utility vs Leverage"):
    """
    Plot Kelly utility as a function of leverage level.

    Args:
        bootstrap_results: List of dicts from run_bootstrap_simulations()
        title: Plot title

    Returns:
        matplotlib figure
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(bootstrap_results)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot Kelly utility vs leverage (using scatter if multiple points per leverage level)
    # or line plot if one point per leverage level
    if df.groupby('leverage').size().max() > 1:
        # Multiple simulations per leverage - use scatter plot
        plt.scatter(df['leverage'], df['kelly_utility'],
                   marker='o', s=50, alpha=0.5, label='Individual simulations')

        # Add mean line
        mean_kelly = df.groupby('leverage')['kelly_utility'].mean().reset_index()
        plt.plot(mean_kelly['leverage'], mean_kelly['kelly_utility'],
                color='darkblue', linewidth=3, alpha=0.8, label='Mean Kelly utility')
    else:
        # One simulation per leverage - use line plot
        plt.plot(df['leverage'], df['kelly_utility'],
                 marker='o', linewidth=2, markersize=6, alpha=0.7, label='Kelly utility')

    plt.xlabel('Leverage Level', fontsize=12)
    plt.ylabel('Kelly Utility', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Find and mark the optimal leverage (based on mean if multiple sims)
    if df.groupby('leverage').size().max() > 1:
        mean_kelly = df.groupby('leverage')['kelly_utility'].mean()
        optimal_lev = mean_kelly.idxmax()
        optimal_kelly = mean_kelly.loc[optimal_lev]
    else:
        optimal_idx = df['kelly_utility'].idxmax()
        optimal_lev = df.loc[optimal_idx, 'leverage']
        optimal_kelly = df.loc[optimal_idx, 'kelly_utility']

    plt.axvline(x=optimal_lev, color='red', linestyle='--', alpha=0.5,
                label=f'Optimal: {optimal_lev:.1f}x (mean Kelly: {optimal_kelly:.2f})')

    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    return plt.gcf()


if __name__ == "__main__":
    # Load and prepare data
    print("Loading data...")
    sp500, fedfunds = load_all_data()
    sp500_adjusted = adjust_returns_for_dividends(sp500, annual_div_yield=0.02)

    # Run bootstrap simulations for Kelly vs Leverage plot
    print("\n" + "="*80)
    print("KELLY UTILITY VS LEVERAGE ANALYSIS")
    print("="*80)
    # Generate leverage levels from 0.0 to 5.0 in steps of 0.1
    leverage_levels = [round(x * 0.1, 1) for x in range(0, 35)]  # 0.0, 0.1, ..., 5.0
    print(f"Running bootstrap simulation for {len(leverage_levels)} leverage levels (0.0x to 5.0x)...")

    bootstrap_results = run_bootstrap_simulations(
        sp500_adjusted,
        fedfunds,
        leverage_levels=leverage_levels,
        n_simulations=10000,
        n_periods=10000,
        random_seed=42
    )

    # Plot Kelly utility vs leverage
    print("\nGenerating Kelly vs Leverage plot...")
    # Calculate simulation stats from results
    df_results = pd.DataFrame(bootstrap_results)
    n_sims = df_results['simulation'].nunique()
    avg_periods = df_results['periods'].iloc[0]  # All should be the same
    # Estimate years: ~252 trading days per year, but we have weekends/holidays
    # so roughly 365 calendar days per year worth of data
    # With 10,000 periods at ~1.05 days per period average = ~10,500 calendar days = ~28.8 years
    n_years = (avg_periods * 1.05) / 365.25  # Rough estimate

    plot_kelly_vs_leverage(bootstrap_results,
                          title=f"Kelly Utility vs Leverage Level\n({n_sims} bootstraps, {int(avg_periods):,} periods, {n_years:.0f} years)")
    plt.savefig('kelly_vs_leverage.png', dpi=300, bbox_inches='tight')
    print("Saved: kelly_vs_leverage.png")

    # Print optimal leverage
    print("\n" + "="*80)
    print("OPTIMAL LEVERAGE")
    print("="*80)
    df_results = pd.DataFrame(bootstrap_results)
    optimal_idx = df_results['kelly_utility'].idxmax()
    optimal_result = df_results.loc[optimal_idx]
    print(f"Optimal leverage: {optimal_result['leverage']:.1f}x")
    print(f"Kelly utility: {optimal_result['kelly_utility']:.4f}")
    print(f"Cumulative return: {optimal_result['cumulative_return']:.2f}x")
    print("="*80)

    # Generate the comparison plot (log scale)
    print("\n" + "="*80)
    print("LEVERAGE COMPARISON PLOT (LOG SCALE)")
    print("="*80)
    print("Generating log scale plot...")
    results_log = plot_leverage_comparison(
        leverage_levels=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        annual_div_yield=0.02,
        margin_spread=0.01,
        initial_value=1000,
        log_scale=True
    )

    # Save log scale version
    plt.savefig('leverage_comparison_log.png', dpi=300, bbox_inches='tight')
    print("Saved: leverage_comparison_log.png")

    # Show all plots
    plt.show()
