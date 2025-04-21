from datetime import timedelta, datetime

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple, Optional, List
import pm4py
import matplotlib.pyplot as plt
from event_object import __BASECOSTS__


def estimate_energy_per_event(energy_df: pd.DataFrame,
                              event_matrix: np.ndarray,
                              event_types: list,
                              timestamp_col: str = 'timestamps',
                              energy_col: str = 'total',
                              include_intercept: bool = True) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Estimates energy consumption per event type using multiple linear regression.

    Parameters:
    -----------
    energy_df : pd.DataFrame
        DataFrame with timestamps, interval indices, and energy values
    event_matrix : np.ndarray
        Matrix where rows are event types and columns are intervals
        Each cell contains the count of events of that type in that interval
    event_types : list
        List of event type names corresponding to rows in event_matrix
    timestamp_col : str, default='timestamps'
        Name of the column containing timestamps in energy_df
    energy_col : str, default='total'
        Name of the column containing energy values in energy_df
    include_intercept : bool, default=True
        Whether to include an intercept term in the regression model

    Returns:
    --------
    Tuple[Dict[str, float], Dict[str, float]]
        - Dictionary mapping event types to their estimated energy consumption
        - Dictionary of model metrics (r2, mse)
    """
    # Ensure we have the same number of intervals in both datasets
    n_intervals = energy_df.shape[0]
    if event_matrix.shape[1] != n_intervals:
        raise ValueError(f"Mismatch between energy_df intervals ({n_intervals}) and "
                         f"event_matrix columns ({event_matrix.shape[1]})")

    # Prepare X (features) and y (target)
    # Transpose event_matrix so columns are event types and rows are intervals
    X = event_matrix.T
    y = energy_df[energy_col].values

    # If requested, add intercept term (column of ones)
    if include_intercept:
        X = np.column_stack((np.ones(X.shape[0]), X))

    # Fit the regression model
    model = LinearRegression(fit_intercept=False)  # We manually add intercept if needed
    model.fit(X, y)

    # Get predictions and model metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    # Create dictionary of coefficients
    coefficients = {}
    if include_intercept:
        coefficients['intercept'] = model.coef_[0]
        for i, event_type in enumerate(event_types):
            coefficients[event_type] = model.coef_[i + 1]
    else:
        for i, event_type in enumerate(event_types):
            coefficients[event_type] = model.coef_[i]

    # Model metrics
    metrics = {
        'r2': r2,
        'mse': mse
    }

    return coefficients, metrics


def analyze_energy_regression(energy_df: pd.DataFrame,
                              event_matrix: np.ndarray,
                              event_types: list,
                              timestamp_col: str = 'timestamps',
                              energy_col: str = 'total') -> pd.DataFrame:
    """
    Performs regression analysis and returns detailed results including
    coefficient estimates, confidence intervals, and p-values.

    Parameters:
    -----------
    energy_df : pd.DataFrame
        DataFrame with timestamps, interval indices, and energy values
    event_matrix : np.ndarray
        Matrix where rows are event types and columns are intervals
    event_types : list
        List of event type names corresponding to rows in event_matrix
    timestamp_col : str, default='timestamps'
        Name of the column containing timestamps in energy_df
    energy_col : str, default='total'
        Name of the column containing energy values in energy_df

    Returns:
    --------
    pd.DataFrame
        DataFrame with detailed regression results
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("This function requires statsmodels. Install with 'pip install statsmodels'")

    # Prepare X (features) and y (target)
    X = event_matrix.T
    y = energy_df[energy_col].values

    # Add intercept
    X = sm.add_constant(X)

    # Fit the regression using statsmodels for detailed statistics
    model = sm.OLS(y, X)
    results = model.fit()

    # Create results dataframe
    param_names = ['intercept'] + event_types
    results_df = pd.DataFrame({
        'coefficient': results.params,
        'std_error': results.bse,
        'p_value': results.pvalues,
        'conf_int_lower': results.conf_int()[0],
        'conf_int_upper': results.conf_int()[1]
    }, index=param_names)

    # Add model summary statistics
    print(f"R-squared: {results.rsquared:.4f}")
    print(f"Adjusted R-squared: {results.rsquared_adj:.4f}")
    print(f"F-statistic: {results.fvalue:.4f}")
    print(f"Prob (F-statistic): {results.f_pvalue:.4f}")

    return results_df


# Example usage:
"""
# Sample energy dataframe
energy_data = {
    'timestamps': pd.date_range(start='2023-01-01', periods=24, freq='1H'),
    'total': np.random.uniform(100, 200, size=24)  # Energy values
}
energy_df = pd.DataFrame(energy_data)

# Sample event matrix
event_types = ['login', 'logout', 'search', 'purchase']
event_matrix = np.random.randint(0, 5, size=(len(event_types), 24))  # Random event counts

# Estimate energy consumption per event type
coefficients, metrics = estimate_energy_per_event(energy_df, event_matrix, event_types)
print("Estimated energy consumption per event type:")
for event_type, energy in coefficients.items():
    print(f"{event_type}: {energy:.2f} units")

print(f"\nModel metrics - R²: {metrics['r2']:.4f}, MSE: {metrics['mse']:.4f}")

# For more detailed analysis (including confidence intervals and p-values)
detailed_results = analyze_energy_regression(energy_df, event_matrix, event_types)
print("\nDetailed regression results:")
print(detailed_results)
"""

def compare_estimated_vs_expected_costs(
        estimated_costs: Dict[str, float],
        expected_costs: Dict[str, float],
        exclude_intercept: bool = True
) -> pd.DataFrame:
    """
    Compares regression-estimated energy costs with expected/known costs for each event type.

    Parameters:
    -----------
    estimated_costs : Dict[str, float]
        Dictionary mapping event types to their estimated energy consumption from regression
    expected_costs : Dict[str, float]
        Dictionary mapping event types to their expected/known energy consumption
    exclude_intercept : bool, default=True
        Whether to exclude the intercept term from comparison

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns for event type, estimated cost, expected cost, 
        difference, and percent difference
    """
    # Create lists to store the data
    event_types = []
    estimated = []
    expected = []
    differences = []
    percent_diffs = []

    # Process each event type
    for event_type in estimated_costs:
        # Skip intercept if requested
        if event_type == 'intercept' and exclude_intercept:
            continue

        # Skip if event type not in expected costs
        if event_type not in expected_costs:
            continue

        est_value = estimated_costs[event_type]
        exp_value = expected_costs[event_type]

        # Calculate difference and percent difference
        diff = est_value - exp_value
        if exp_value != 0:
            percent_diff = (diff / exp_value) * 100
        else:
            percent_diff = float('inf') if diff != 0 else 0

        # Add to lists
        event_types.append(event_type)
        estimated.append(est_value)
        expected.append(exp_value)
        differences.append(diff)
        percent_diffs.append(percent_diff)

    # Create and return dataframe
    comparison_df = pd.DataFrame({
        'event_type': event_types,
        'estimated_cost': estimated,
        'expected_cost': expected,
        'difference': differences,
        'percent_difference': percent_diffs
    })

    return comparison_df


def visualize_cost_comparison(comparison_df: pd.DataFrame,
                              title: str = "Estimated vs Expected Energy Costs per Event Type") -> None:
    """
    Creates a visual comparison of estimated vs expected costs.

    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame from compare_estimated_vs_expected_costs function
    title : str, default="Estimated vs Expected Energy Costs per Event Type"
        Title for the plot
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style
        sns.set_style("whitegrid")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get event types and positions
        event_types = comparison_df['event_type']
        x = np.arange(len(event_types))
        width = 0.35

        # Create bars
        rects1 = ax.bar(x - width / 2, comparison_df['estimated_cost'], width, label='Estimated Cost')
        rects2 = ax.bar(x + width / 2, comparison_df['expected_cost'], width, label='Expected Cost')

        # Add details
        ax.set_title(title)
        ax.set_xlabel('Event Type')
        ax.set_ylabel('Energy Cost')
        ax.set_xticks(x)
        ax.set_xticklabels(event_types, rotation=45, ha='right')
        ax.legend()

        # Add value labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        # Add percentage difference text
        for i, event_type in enumerate(event_types):
            percent_diff = comparison_df.loc[comparison_df['event_type'] == event_type, 'percent_difference'].values[0]
            y_pos = max(comparison_df.loc[comparison_df['event_type'] == event_type, 'estimated_cost'].values[0],
                        comparison_df.loc[comparison_df['event_type'] == event_type, 'expected_cost'].values[0])
            ax.annotate(f'{percent_diff:.1f}%',
                        xy=(i, y_pos),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        color='red' if abs(percent_diff) > 10 else 'green')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Visualization requires matplotlib and seaborn. Install with 'pip install matplotlib seaborn'")
        print("\nComparison results:")
        print(comparison_df)


def estimate_and_compare_costs(energy_df: pd.DataFrame,
                               event_matrix: np.ndarray,
                               event_types: List[str],
                               expected_costs: Dict[str, float],
                               timestamp_col: str = 'timestamps',
                               energy_col: str = 'total',
                               visualize: bool = True) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    End-to-end function that estimates energy costs via regression and compares them with expected costs.

    Parameters:
    -----------
    energy_df : pd.DataFrame
        DataFrame with timestamps, interval indices, and energy values
    event_matrix : np.ndarray
        Matrix where rows are event types and columns are intervals
    event_types : List[str]
        List of event type names corresponding to rows in event_matrix
    expected_costs : Dict[str, float]
        Dictionary mapping event types to their expected/known energy consumption
    timestamp_col : str, default='timestamps'
        Name of the column containing timestamps in energy_df
    energy_col : str, default='total'
        Name of the column containing energy values in energy_df
    visualize : bool, default=True
        Whether to generate and display a visualization of the comparison

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, float]]
        - DataFrame with comparison results
        - Dictionary with model metrics
    """
    # Run regression to estimate costs
    estimated_costs, metrics = estimate_energy_per_event(
        energy_df, event_matrix, event_types,
        timestamp_col=timestamp_col, energy_col=energy_col
    )

    # Compare with expected costs
    comparison_df = compare_estimated_vs_expected_costs(estimated_costs, expected_costs)

    # Calculate overall comparison metrics
    metrics['mean_abs_diff'] = comparison_df['difference'].abs().mean()
    metrics['mean_abs_percent_diff'] = comparison_df['percent_difference'].abs().mean()

    # Generate visualization if requested
    if visualize:
        visualize_cost_comparison(comparison_df)

    return comparison_df, metrics


# Example usage:
"""
# Sample energy dataframe
energy_data = {
    'timestamps': pd.date_range(start='2023-01-01', periods=24, freq='1H'),
    'total': np.random.uniform(100, 200, size=24)  # Energy values
}
energy_df = pd.DataFrame(energy_data)

# Sample event matrix
event_types = ['login', 'logout', 'search', 'purchase']
event_matrix = np.random.randint(0, 5, size=(len(event_types), 24))  # Random event counts

# Expected costs per event type
expected_costs = {
    'login': 15.0,
    'logout': 10.0,
    'search': 25.0,
    'purchase': 50.0
}

# Run comparison
comparison_df, metrics = estimate_and_compare_costs(energy_df, event_matrix, event_types, expected_costs)

print("\nComparison Results:")
print(comparison_df)

print("\nModel and Comparison Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
"""


def create_event_matrix(xes_path, interval_length, start_timestamp=None, end_timestamp=None):
    """
    Creates a matrix where rows represent event types and columns represent time intervals.
    Each cell contains the count of events of a specific type during a specific time interval.

    Parameters:
    -----------
    xes_path : str
        Path to the XES file
    interval_length : int
        Length of each interval in seconds
    start_timestamp : datetime, optional
        Start timestamp for analysis (if None, uses the earliest event timestamp)
    end_timestamp : datetime, optional
        End timestamp for analysis (if None, uses the latest event timestamp)

    Returns:
    --------
    numpy.ndarray
        Matrix of event counts
    list
        List of event types (row labels)
    list
        List of interval start times (column labels)
    """
    # Import the XES file
    log = pm4py.read_xes(xes_path)

    # Convert to DataFrame for easier manipulation
    df = pm4py.convert_to_dataframe(log)

    # Ensure we have timestamp and concept:name columns (standard XES attributes)
    if 'time:timestamp' not in df.columns or 'concept:name' not in df.columns:
        raise ValueError("XES file must contain 'time:timestamp' and 'concept:name' attributes")

    # Ensure timestamp is in datetime format
    if not isinstance(df['time:timestamp'].iloc[0], datetime):
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

    # Determine start and end timestamps if not provided
    if start_timestamp is None:
        start_timestamp = df['time:timestamp'].min()
    elif isinstance(start_timestamp, str):
        start_timestamp = pd.to_datetime(start_timestamp)

    if end_timestamp is None:
        end_timestamp = df['time:timestamp'].max()
    elif isinstance(end_timestamp, str):
        end_timestamp = pd.to_datetime(end_timestamp)

    # Filter events within the specified time range
    filtered_df = df[(df['time:timestamp'] >= start_timestamp) &
                     (df['time:timestamp'] <= end_timestamp)]

    if filtered_df.empty:
        raise ValueError("No events found in the specified time range")

    # Get unique event types
    event_types = filtered_df['concept:name'].unique()

    # Create time intervals
    interval_delta = timedelta(seconds=interval_length)
    current_time = start_timestamp
    intervals = []

    while current_time <= end_timestamp:
        intervals.append(current_time)
        current_time += interval_delta

    # Create an empty matrix
    matrix = np.zeros((len(event_types), len(intervals)))

    # Fill the matrix with event counts
    for i, event_type in enumerate(event_types):
        for j, interval_start in enumerate(intervals[:-1]):
            interval_end = intervals[j + 1]
            count = len(filtered_df[(filtered_df['concept:name'] == event_type) &
                                    (filtered_df['time:timestamp'] >= interval_start) &
                                    (filtered_df['time:timestamp'] < interval_end)])
            matrix[i, j] = count

        # Handle the last interval separately to include the end_timestamp
        if len(intervals) > 1:
            last_interval_start = intervals[-1]
            count = len(filtered_df[(filtered_df['concept:name'] == event_type) &
                                    (filtered_df['time:timestamp'] >= last_interval_start) &
                                    (filtered_df['time:timestamp'] <= end_timestamp)])
            matrix[i, -1] = count

    return matrix, list(event_types), intervals


if __name__ == '__main__':
    with open('X:/PHD/lehner_phd_overview/signal_generator/experiment_1_1_1_30min.csv', 'r') as file:
        df = pd.read_csv(file)

    # create event matrix from corresponding csv
    event_matrix, event_types, intervals = create_event_matrix(
        xes_path='X:/PHD/lehner_phd_overview/signal_generator/filtered_log.xes',
        interval_length=1800,  # 30 minutes
    )
    with open('event_matrix_30min.csv',"w") as writer:
        pd.DataFrame(event_matrix).to_csv(writer, index=False)
    # Assuming the CSV has columns 'timestamps' and 'total'
    report = estimate_and_compare_costs(df, event_matrix, event_types, __BASECOSTS__, visualize=True)
    print(report)

    with open('report.txt', 'wb') as file:
        file.write(report.encode('utf-8'))