"""
Visualization functions for creating plots and charts.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import traceback
from backend.config.app_config import CONFIG
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Configure logging
from backend.logs.logger_setup import setup_logger


logger = setup_logger('visualization', 'visualization.log')

# Set default visualization style
sns.set_theme(style=CONFIG["visualization"]["style"])

# Define color palette based on config
COLORS = CONFIG["visualization"].get("colors", {
    "primary": "#1E88E5",
    "secondary": "#FFC107",
    "success": "#4CAF50",
    "danger": "#F44336",
    "warning": "#FF9800"
})

def format_thousands(x, pos):
    """Format y-axis labels to show thousands with K suffix"""
    if x >= 1000:
        return f'{x/1000:.1f}K'
    return f'{x:.0f}'

def create_figure(width=None, height=None):
    """Create a figure with the specified dimensions"""
    width = width or CONFIG["visualization"]["default_width"]
    height = height or CONFIG["visualization"]["default_height"]
    return plt.figure(figsize=(width, height), dpi=CONFIG["visualization"]["dpi"])









def create_hourly_block_consumption_plot(df, plant_name, selected_date):
    """
    Create a bar plot of hourly block consumption data (Time-of-Day Hourly Consumption Trend)

    Args:
        df (DataFrame): Hourly block consumption data
        plant_name (str): Name of the plant
        selected_date (datetime): Selected date for the plot

    Returns:
        Figure: Matplotlib figure object
    """
    try:
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Format the hour blocks for display
        df['HOUR_BLOCK_LABEL'] = df['HOUR_BLOCK'].apply(lambda x: f"{int(x):02d}:00 - {int(x)+3:02d}:00")

        # Plot the data with a softer color for ToD Consumption
        bars = sns.barplot(
            data=df,
            x='HOUR_BLOCK_LABEL',
            y='TOTAL_CONSUMPTION',
            color=COLORS.get("consumption", "#00897B"),  # Teal color for consumption
            alpha=0.8,  # Add transparency
            ax=ax
        )

        # Add data labels on top of bars
        for bar in bars.patches:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 5,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

        # Customize the plot
        date_str = selected_date.strftime('%Y-%m-%d')
        ax.set_title(f"ToD Consumption for {plant_name} on {date_str}", fontsize=16, pad=20)
        ax.set_ylabel("Total Consumption (kWh)", fontsize=12)
        ax.set_xlabel("Time Block", fontsize=12)

        # Format y-axis with K for thousands
        ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')

        # Add grid for y-axis only
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)  # Lighter grid

        # Add average line
        if not df.empty:
            avg = df['TOTAL_CONSUMPTION'].mean()
            ax.axhline(
                y=avg,
                color=COLORS.get("average", "#757575"),  # Gray for average
                linestyle='--',
                linewidth=1.5,
                label=f'Average: {avg:.1f}'
            )
            ax.legend(loc='upper right', frameon=True, framealpha=0.9)

        # Add subtle watermark
        fig.text(0.99, 0.01, 'ToD Consumption',
                fontsize=8, color='gray', ha='right', va='bottom', alpha=0.5)  # More subtle

        # Adjust layout
        plt.tight_layout()
        return fig

    except Exception as e:
        logger.error(f"Error creating hourly block consumption plot: {e}")
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}",
                ha='center', va='center', fontsize=12)
        return fig



def create_consumption_plot(df, plant_name):
    """
    Create a line plot of consumption data with improved styling

    Args:
        df (DataFrame): Consumption data
        plant_name (str): Name of the plant

    Returns:
        Figure: Matplotlib figure object
    """
    try:
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Check if we have hourly data or daily data
        if 'hour' in df.columns and 'energy_kwh' in df.columns:
            # We have hourly data - create a bar chart
            # Create hour labels for x-axis
            df = df.copy()
            df['hour_label'] = df['hour'].apply(lambda x: f"{int(x):02d}:00")

            # Plot the data with a softer color for consumption
            bars = sns.barplot(
                data=df,
                x='hour_label',
                y='energy_kwh',
                color=COLORS.get("consumption", "#00897B"),  # Teal color (easier on eyes)
                alpha=0.8,  # Add transparency
                ax=ax
            )

            # Add data labels on top of bars
            for i, bar in enumerate(ax.patches):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 5,
                    f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

            # Customize the plot
            ax.set_title(f"Consumption for {plant_name}", fontsize=16, pad=20)
            ax.set_ylabel("Consumption (kWh)", fontsize=12)
            ax.set_xlabel("Hour", fontsize=12)

            # Add average consumption line
            if not df.empty:
                avg = df['energy_kwh'].mean()
                ax.axhline(
                    y=avg,
                    color=COLORS.get("average", "#757575"),  # Gray for average
                    linestyle='--',
                    linewidth=1.5,
                    label=f'Average: {avg:.1f}'
                )
                ax.legend(loc='upper right', frameon=True, framealpha=0.9)
        else:
            # We need to aggregate the data by date
            # Check if we have a date column
            if 'date' not in df.columns:
                # Create a dummy date column for demonstration
                import pandas as pd
                from datetime import datetime, timedelta

                # Create a date range for the last 7 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=6)
                date_range = pd.date_range(start=start_date, end=end_date)

                # Create a new DataFrame with dates and random consumption values
                import numpy as np
                consumption_values = np.random.randint(100, 500, size=len(date_range))
                date_df = pd.DataFrame({
                    'DATE': date_range,
                    'CONSUMPTION': consumption_values
                })

                # Use this DataFrame for plotting
                df = date_df

            # Plot the data
            sns.lineplot(
                data=df,
                x='DATE',
                y='CONSUMPTION',
                marker='o',
                markersize=6,
                linewidth=2,
                color=COLORS.get("consumption", "#00897B"),  # Teal color
                alpha=0.9,  # Slight transparency
                ax=ax
            )

            # No moving average line

            # Import the helper function to get plant display name
            from backend.data.data import get_plant_display_name

            # Get the display name for the plant
            plant_display_name = get_plant_display_name(plant_name)

            # Customize the plot
            ax.set_title(f"Consumption for {plant_display_name}", fontsize=16, pad=20)
            ax.set_ylabel("Consumption (kWh)", fontsize=12)
            ax.set_xlabel("Date", fontsize=12)

            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            if len(df) > 30:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            else:
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))

            # Add annotations for max and min values
            if not df.empty:
                max_cons = df['CONSUMPTION'].max()
                max_date = df.loc[df['CONSUMPTION'].idxmax(), 'DATE']
                min_cons = df['CONSUMPTION'].min()
                min_date = df.loc[df['CONSUMPTION'].idxmin(), 'DATE']

                # Only annotate if we have enough data points
                if len(df) > 5:
                    ax.annotate(f'Max: {max_cons:.1f}',
                                xy=(max_date, max_cons),
                                xytext=(0, 15),
                                textcoords='offset points',
                                ha='center',
                                va='bottom',
                                fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

                    ax.annotate(f'Min: {min_cons:.1f}',
                                xy=(min_date, min_cons),
                                xytext=(0, -15),
                                textcoords='offset points',
                                ha='center',
                                va='top',
                                fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

        # Add subtle watermark
        fig.text(0.99, 0.01, 'Consumption Data',
                fontsize=8, color='gray', ha='right', va='bottom', alpha=0.5)  # More subtle

        # Format y-axis with K for thousands
        ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)  # Lighter grid

        # Adjust layout
        plt.tight_layout()
        return fig

    except Exception as e:
        logger.error(f"Error creating consumption plot: {e}")
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}",
                ha='center', va='center', fontsize=12)
        return fig


def create_comparison_plot(df, plant_name, date):
    """Create a plot comparing generation and consumption data with surplus calculations"""
    sns.set_theme(style="whitegrid")

    # Sort by time_interval to ensure proper ordering (supports both 15-min and hourly data)
    time_col = 'time_interval' if 'time_interval' in df.columns else 'hour'
    df = df.sort_values(time_col)

    # Determine which column name is used for consumption
    consumption_col = 'consumption_kwh' if 'consumption_kwh' in df.columns else 'energy_kwh'

    # Calculate surplus generation and demand
    df['surplus_generation'] = df.apply(lambda row: max(0, row['generation_kwh'] - row[consumption_col]), axis=1)
    df['surplus_demand'] = df.apply(lambda row: max(0, row[consumption_col] - row['generation_kwh']), axis=1)

    # Calculate totals for annotation
    total_generation = df['generation_kwh'].sum()
    total_consumption = df[consumption_col].sum()
    total_surplus_gen = df['surplus_generation'].sum()
    total_surplus_demand = df['surplus_demand'].sum()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot generation (solid green line)
    ax.plot(df[time_col], df['generation_kwh'], color='green', linewidth=3,
            marker='o', markersize=6, label='Generation')

    # Plot consumption (dashed red line)
    ax.plot(df[time_col], df[consumption_col], color='red', linewidth=3,
            linestyle='--', marker='s', markersize=6, label='Consumption')

    # Add transparent fill between curves
    for i in range(len(df)-1):
        time_range = [df.iloc[i][time_col], df.iloc[i+1][time_col]]
        gen_vals = [df.iloc[i]['generation_kwh'], df.iloc[i+1]['generation_kwh']]
        cons_vals = [df.iloc[i][consumption_col], df.iloc[i+1][consumption_col]]

        fill_color = 'green' if gen_vals[0] > cons_vals[0] else 'red'
        ax.fill_between(time_range, gen_vals, cons_vals, color=fill_color, alpha=0.2)

    # Axes labels and ticks - adjust based on data granularity
    if time_col == 'time_interval':
        # For 15-minute data, show every 2 hours (8 intervals)
        ax.set_xticks([i for i in range(0, 24, 2)])
        ax.set_xlabel("Time of Day (15-minute intervals)")
    else:
        # For hourly data
        ax.set_xticks(range(0, 24))
        ax.set_xlabel("Hour of Day")

    ax.set_ylabel("Energy (kWh)")

    # Summary text box removed as requested

    ax.legend(loc='upper right')

    # Import the helper function to get plant display name
    from backend.data.data import get_plant_display_name

    # Get the display name for the plant
    plant_display_name = get_plant_display_name(plant_name)

    # Title and layout
    plt.title(f"Energy Generation vs Consumption for {plant_display_name} on {date.strftime('%Y-%m-%d')}")
    plt.tight_layout()

    return fig

def create_daily_consumption_plot(df, plant_name, start_date, end_date):
    """
    Create a line plot of hourly consumption data
    """
    try:
        # Standardize column names to lowercase
        df.columns = [col.lower() for col in df.columns]

        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Check if we have the datetime column for hourly plotting
        if 'datetime' in df.columns:
            # Create a copy of the dataframe for plotting
            plot_df = df.copy()

            # Sort by datetime to ensure chronological order
            plot_df = plot_df.sort_values('datetime')

            # Plot the data with hourly granularity
            sns.lineplot(
                data=plot_df,
                x='datetime',
                y='consumption_kwh',
                marker='o',
                markersize=4,
                linewidth=2,
                color=COLORS.get("consumption", "#00897B"),  # Teal color
                alpha=0.9,
                ax=ax
            )

            # Format x-axis for hourly data
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

            # Set appropriate x-axis tick frequency based on date range
            days_diff = (end_date - start_date).days
            if days_diff <= 1:
                # For single day, show every 2 hours
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            elif days_diff <= 3:
                # For up to 3 days, show every 6 hours
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            elif days_diff <= 7:
                # For up to a week, show every 12 hours
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            else:
                # For longer periods, show daily at noon
                ax.xaxis.set_major_locator(mdates.DayLocator())
                # Add minor ticks for hours
                ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
                ax.grid(True, which='minor', axis='x', linestyle=':', alpha=0.2)

        else:
            # Fallback to daily plot if datetime column is not available
            sns.lineplot(
                data=df,
                x='date',
                y='consumption_kwh',
                marker='o',
                markersize=6,
                linewidth=2,
                color=COLORS.get("consumption", "#00897B"),  # Teal color
                alpha=0.9,
                ax=ax
            )

            # Format x-axis for daily data
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator())

        # Import the helper function to get plant display name
        from backend.data.data import get_plant_display_name

        # Get the display name for the plant
        plant_display_name = get_plant_display_name(plant_name)

        # Customize the plot
        date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        ax.set_title(f"Hourly Consumption for {plant_display_name} ({date_range})", fontsize=16, pad=20)
        ax.set_ylabel("Consumption (kWh)", fontsize=12)
        ax.set_xlabel("Time", fontsize=12)

        # Format y-axis with K for thousands
        ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')

        # Add grid
        ax.grid(True, axis='both', linestyle='--', alpha=0.5)

        # Add subtle watermark
        fig.text(0.99, 0.01, 'Hourly Consumption',
                fontsize=8, color='gray', ha='right', va='bottom', alpha=0.5)

        # Adjust layout
        plt.tight_layout()
        return fig

    except Exception as e:
        logger.error(f"Error creating daily consumption plot: {e}")
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}",
                ha='center', va='center', fontsize=12)
        return fig

def create_daily_comparison_plot(df, plant_name, start_date, end_date):
    """Create a plot comparing daily generation and consumption data with surplus calculations"""
    sns.set_theme(style="whitegrid")

    try:
        # Standardize column names to lowercase
        df.columns = [col.lower() for col in df.columns]

        # Calculate surplus generation and demand
        df['surplus_generation'] = df.apply(lambda row: max(0, row['generation_kwh'] - row['consumption_kwh']), axis=1)
        df['surplus_demand'] = df.apply(lambda row: max(0, row['consumption_kwh'] - row['generation_kwh']), axis=1)

        # Calculate totals for annotation
        total_generation = df['generation_kwh'].sum()
        total_consumption = df['consumption_kwh'].sum()
        total_surplus_gen = df['surplus_generation'].sum()
        total_surplus_demand = df['surplus_demand'].sum()

        # Create the figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})

        # Top plot: Generation vs Consumption
        # Plot generation (solid green line)
        ax1.plot(df['date'], df['generation_kwh'], color='green', linewidth=3,
                marker='o', markersize=8, label='Generation')

        # Plot consumption (dashed red line) - handle zero consumption gracefully
        # Separate zero and non-zero consumption for better visualization
        zero_consumption_mask = df['consumption_kwh'] == 0
        non_zero_consumption = df[~zero_consumption_mask]
        zero_consumption = df[zero_consumption_mask]

        # Plot non-zero consumption with normal style
        if not non_zero_consumption.empty:
            ax1.plot(non_zero_consumption['date'], non_zero_consumption['consumption_kwh'],
                    color='red', linewidth=3, linestyle='--', marker='s', markersize=8,
                    label='Consumption')

        # Plot zero consumption points with different style (smaller markers, different color)
        if not zero_consumption.empty:
            ax1.scatter(zero_consumption['date'], zero_consumption['consumption_kwh'],
                       color='orange', marker='x', s=100, linewidth=3,
                       label='Zero Consumption', alpha=0.8)

        # Add transparent fill between curves (only for non-zero consumption)
        if len(non_zero_consumption) > 1:
            for i in range(len(non_zero_consumption)-1):
                date_range = [non_zero_consumption.iloc[i]['date'], non_zero_consumption.iloc[i+1]['date']]
                gen_vals = [non_zero_consumption.iloc[i]['generation_kwh'], non_zero_consumption.iloc[i+1]['generation_kwh']]
                cons_vals = [non_zero_consumption.iloc[i]['consumption_kwh'], non_zero_consumption.iloc[i+1]['consumption_kwh']]

                fill_color = 'green' if gen_vals[0] > cons_vals[0] else 'red'
                ax1.fill_between(date_range, gen_vals, cons_vals, color=fill_color, alpha=0.2)

        # Format x-axis dates for top plot
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        if (end_date - start_date).days > 30:
            ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        else:
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))

        # Format y-axis with K for thousands for top plot
        ax1.yaxis.set_major_formatter(FuncFormatter(format_thousands))

        # Axes labels for top plot
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Energy (kWh)")

        # Add legend to top plot
        ax1.legend(loc='upper right')

        # Rotate x-axis labels for top plot
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # Summary text box removed as requested

        # Bottom plot: Surplus Generation and Demand
        # Plot surplus generation (green bars)
        ax2.bar(df['date'], df['surplus_generation'], color='green', alpha=0.6, label='Surplus Generation')

        # Plot surplus demand (red bars)
        ax2.bar(df['date'], df['surplus_demand'], color='red', alpha=0.6, label='Surplus Demand')

        # Format x-axis dates for bottom plot
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        if (end_date - start_date).days > 30:
            ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        else:
            ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))

        # Format y-axis with K for thousands for bottom plot
        ax2.yaxis.set_major_formatter(FuncFormatter(format_thousands))

        # Axes labels for bottom plot
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Energy (kWh)")

        # Add legend to bottom plot
        ax2.legend(loc='upper right')

        # Rotate x-axis labels for bottom plot
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # Add grid to bottom plot
        ax2.grid(True, linestyle='--', alpha=0.5)

        # Import the helper function to get plant display name
        from backend.data.data import get_plant_display_name

        # Get the display name for the plant
        plant_display_name = get_plant_display_name(plant_name)

        # Title and layout
        date_range_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        fig.suptitle(f"Daily Energy Generation vs Consumption for {plant_display_name} ({date_range_str})",
                    fontsize=16, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.3)  # Adjust spacing between subplots

        return fig

    except Exception as e:
        logger.error(f"Error creating daily comparison plot: {e}")
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}",
                ha='center', va='center', fontsize=12)
        return fig



def create_tod_binned_plot(df, plant_name, start_date, end_date=None):
    """
    Create a bar plot comparing generation vs consumption with custom ToD bins

    Args:
        df (DataFrame): ToD binned data with generation and consumption
        plant_name (str): Name of the plant
        start_date (datetime): Start date
        end_date (datetime, optional): End date. If None, only start_date is used.

    Returns:
        Figure: Matplotlib figure object
    """

    try:
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 7))

        # Set width of bars
        bar_width = 0.35

        # Set positions of the bars on X axis
        r1 = np.arange(len(df))
        r2 = [x + bar_width for x in r1]

        # Create bars
        generation_bars = ax.bar(
            r1,
            df['generation_kwh'],
            width=bar_width,
            label='Generation',
            color=COLORS.get("generation", "#4CAF50"),
            alpha=0.8
        )

        consumption_bars = ax.bar(
            r2,
            df['consumption_kwh'],
            width=bar_width,
            label='Consumption',
            color=COLORS.get("consumption", "#F44336"),
            alpha=0.8
        )

        # Add data labels on top of bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 5,
                    f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

        add_labels(generation_bars)
        add_labels(consumption_bars)

        # Add peak/off-peak background shading
        for i, is_peak in enumerate(df['is_peak']):
            if is_peak:
                # Light yellow background for peak periods
                ax.axvspan(i - 0.4, i + 0.8, alpha=0.2, color='#FFF9C4')

        # Import the helper function to get plant display name
        from backend.data.data import get_plant_display_name

        # Get the display name for the plant
        plant_display_name = get_plant_display_name(plant_name)

        # Add labels and title
        if end_date is None or start_date == end_date:
            date_str = start_date.strftime('%Y-%m-%d')
            ax.set_title(f"ToD Generation vs Consumption for {plant_display_name} on {date_str}", fontsize=16, pad=20)
        else:
            date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            ax.set_title(f"ToD Generation vs Consumption for {plant_display_name} ({date_range})", fontsize=16, pad=20)

        ax.set_ylabel("Energy (kWh)", fontsize=12)
        ax.set_xlabel("Time of Day", fontsize=12)

        # Set x-axis ticks
        ax.set_xticks([r + bar_width/2 for r in range(len(df))])
        ax.set_xticklabels(df['tod_bin'])

        # Format y-axis with K for thousands
        ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Add grid for y-axis only
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        # Add legend
        ax.legend(loc='upper right', frameon=True, framealpha=0.9)

        # Add annotations for peak/off-peak periods
        ax.annotate(
            'Peak Periods',
            xy=(0.02, 0.97),
            xycoords='axes fraction',
            backgroundcolor='#FFF9C4',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF9C4", ec="gray", alpha=0.8)
        )

        # Calculate and display replacement percentages
        for i in range(len(df)):
            gen = df['generation_kwh'].iloc[i]
            cons = df['consumption_kwh'].iloc[i]
            if cons > 0:
                # Calculate raw replacement percentage
                raw_replacement = (gen / cons) * 100
                # Cap at 100% for display purposes
                replacement = min(100, raw_replacement)

                # Log both values for debugging
                logger.info(f"ToD bin {df['tod_bin'].iloc[i]} - Raw replacement %: {raw_replacement:.2f}%, Capped: {replacement:.2f}%")

                ax.text(
                    i + bar_width/2,
                    max(gen, cons) + 10,
                    f"{replacement:.1f}%",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    color='#1565C0'
                )

        # Add subtle watermark
        fig.text(0.99, 0.01, 'ToD Analysis',
                fontsize=8, color='gray', ha='right', va='bottom', alpha=0.5)

        # Adjust layout
        plt.tight_layout()
        return fig

    except Exception as e:
        logger.error(f"Error creating ToD binned plot: {e}")
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}",
                ha='center', va='center', fontsize=12)
        return fig


def create_generation_only_plot(df, plant_name, start_date, end_date=None):
    """
    Create a generation-only plot (line chart for single day, bar chart for date ranges).

    Args:
        df (DataFrame): Generation data with 'time' and 'generation_kwh' columns
        plant_name (str): Name of the plant
        start_date (datetime): Start date
        end_date (datetime, optional): End date. If None, only start_date is used.

    Returns:
        Figure: Matplotlib figure object
    """
    try:
        # Determine if single day or date range
        is_single_day = end_date is None or start_date == end_date

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        if is_single_day:
            # Line chart for single day (hourly data)
            if 'hour' in df.columns:
                # Use hour for x-axis
                ax.plot(df['hour'], df['generation_kwh'],
                       color=COLORS.get("primary", "#4285F4"),
                       marker='o', markersize=6, linewidth=2)
                ax.set_xlabel("Hour of Day", fontsize=12)
                ax.set_xlim(0, 23)
                ax.set_xticks(range(0, 24, 2))
            else:
                # Use time for x-axis
                ax.plot(df['time'], df['generation_kwh'],
                       color=COLORS.get("primary", "#4285F4"),
                       marker='o', markersize=6, linewidth=2)
                ax.set_xlabel("Time", fontsize=12)

            title = f"Generation - {plant_name}\n{start_date.strftime('%B %d, %Y')}"

        else:
            # Bar chart for date ranges (daily data)
            ax.bar(df['time'], df['generation_kwh'],
                  color=COLORS.get("primary", "#4285F4"),
                  alpha=0.8, width=0.8)
            ax.set_xlabel("Date", fontsize=12)

            # Format x-axis dates
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            if (end_date - start_date).days > 30:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            else:
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))

            # Rotate x-axis labels for better readability
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            title = f"Generation - {plant_name}\n{start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}"

        # Common formatting
        ax.set_ylabel("Generation (kWh)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Format y-axis with K for thousands
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        return fig

    except Exception as e:
        logger.error(f"Error creating generation-only plot: {e}")
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}",
                ha='center', va='center', fontsize=12)
        return fig


def create_tod_generation_plot(df, plant_name, start_date, end_date=None):
    """
    Create a stacked bar chart of generation data with custom ToD bins
    based on the configuration settings.

    Args:
        df (DataFrame): ToD binned data with generation
        plant_name (str): Name of the plant
        start_date (datetime): Start date
        end_date (datetime, optional): End date. If None, only start_date is used.

    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    try:
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 7))

        # Define ToD categories for stacking
        tod_categories = {
            0: '10 AM - 6 PM (Off-Peak)',
            1: '10 PM - 6 AM (Off-Peak)',
            2: '6 AM - 10 AM (Peak)',
            3: '6 PM - 10 PM (Peak)'
        }

        # Define colors for each ToD category - using more distinct colors
        tod_colors = {
            '10 AM - 6 PM (Off-Peak)': '#FFC107',  # Amber/Yellow for daytime off-peak
            '10 PM - 6 AM (Off-Peak)': '#3F51B5',  # Indigo for nighttime off-peak
            '6 AM - 10 AM (Peak)': '#FF5722',      # Deep Orange for morning peak
            '6 PM - 10 PM (Peak)': '#E91E63'       # Pink for evening peak
        }

        is_single_day = end_date is None or start_date == end_date

        if is_single_day:
            # For single day view, create a stacked bar chart with ToD categories

            # Check if we have the right data structure
            if 'tod_bin' in df.columns and 'generation_kwh' in df.columns:
                # Create a mapping from existing tod_bin to our categories
                # This assumes the tod_bin format matches what's in the data

                # Create a new dataframe with the data organized by our ToD categories
                stacked_data = {}

                # Initialize with zeros
                for date_val in df['date'].unique() if 'date' in df.columns else [start_date]:
                    stacked_data[date_val] = {cat: 0 for cat in tod_categories.values()}

                # Fill in the values from our data
                for _, row in df.iterrows():
                    tod_bin = row['tod_bin']
                    gen_kwh = row['generation_kwh']
                    date_val = row['date'] if 'date' in df.columns else start_date

                    # Find the matching category
                    for _, cat in tod_categories.items():
                        if cat in tod_bin or tod_bin in cat:
                            stacked_data[date_val][cat] = gen_kwh
                            break

                # Convert to DataFrame for plotting
                plot_data = pd.DataFrame(stacked_data).T

                # Create the stacked bar chart
                bottom = np.zeros(len(plot_data))

                # Plot each category as a segment of the stacked bar
                for cat in tod_categories.values():
                    if cat in plot_data.columns:
                        ax.bar(
                            range(len(plot_data)),
                            plot_data[cat],
                            bottom=bottom,
                            label=cat,
                            color=tod_colors[cat],
                            alpha=0.8,
                            width=0.6
                        )
                        bottom += plot_data[cat].values

                # Add total value on top of the stacked bar
                for i, total in enumerate(plot_data.sum(axis=1)):
                    ax.text(
                        i,
                        total + 5,
                        f'{total:.1f}',
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        fontweight='bold'
                    )

                # Set x-axis labels
                if 'date' in df.columns:
                    ax.set_xticks(range(len(plot_data)))
                    ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in plot_data.index])
                else:
                    ax.set_xticks([0])
                    ax.set_xticklabels([start_date.strftime('%Y-%m-%d')])
            else:
                # Fallback to simple bar chart if data structure doesn't match
                logger.warning("Data structure doesn't match expected format for stacked bar chart")
                bars = ax.bar(
                    np.arange(len(df)),
                    df['generation_kwh'],
                    width=0.6,
                    color=COLORS.get("generation", "#4CAF50"),
                    alpha=0.8
                )

                # Add data labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 5,
                        f'{height:.1f}',
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )

                # Set x-axis ticks
                ax.set_xticks(np.arange(len(df)))
                ax.set_xticklabels(df['tod_bin'])
        else:
            # For multiple days view, create a stacked bar chart for each day

            # Check if we have date information
            if 'date' in df.columns:
                # Sort the dataframe by date
                df = df.sort_values('date')

                # Get unique dates
                unique_dates = df['date'].unique()
                num_dates = len(unique_dates)

                # Create a mapping from tod_bin to our categories
                tod_bin_to_category = {}
                for _, row in df.iterrows():
                    tod_bin = row['tod_bin']
                    for _, cat in tod_categories.items():
                        if cat in tod_bin or tod_bin in cat:
                            tod_bin_to_category[tod_bin] = cat
                            break

                # Create a new dataframe with data organized by date and ToD category
                plot_data = []

                # Process each date
                for date_val in unique_dates:
                    date_df = df[df['date'] == date_val]

                    # Initialize data for this date with zeros for all categories
                    date_data = {
                        'date': date_val,
                        'date_str': date_val.strftime('%Y-%m-%d')
                    }

                    # Initialize all categories with zero
                    for cat in tod_categories.values():
                        date_data[cat] = 0

                    # Fill in the values from our data
                    for _, row in date_df.iterrows():
                        tod_bin = row['tod_bin']
                        gen_kwh = row['generation_kwh']

                        # Find the matching category
                        if tod_bin in tod_bin_to_category:
                            cat = tod_bin_to_category[tod_bin]
                            date_data[cat] = gen_kwh

                    plot_data.append(date_data)

                # Convert to DataFrame for plotting
                plot_df = pd.DataFrame(plot_data)

                # Set up x positions for the bars
                x = np.arange(num_dates)
                width = 0.6

                # Create the stacked bar chart - one stacked bar for each date
                bottom = np.zeros(num_dates)

                # Plot each category as a segment of the stacked bars
                for cat in tod_categories.values():
                    if cat in plot_df.columns:
                        ax.bar(
                            x,
                            plot_df[cat],
                            bottom=bottom,
                            label=cat,
                            color=tod_colors[cat],
                            alpha=0.8,
                            width=width
                        )
                        bottom += plot_df[cat].values

                # Add total value on top of each stacked bar
                for i, (_, row) in enumerate(plot_df.iterrows()):
                    total = sum(row[cat] for cat in tod_categories.values() if cat in row)
                    ax.text(
                        i,
                        total + 5,
                        f'{total:.1f}',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold'
                    )

                # Set x-axis labels
                ax.set_xticks(x)
                ax.set_xticklabels(plot_df['date_str'], rotation=45, ha='right')
            else:
                # Fallback to simple bar chart if no date information
                logger.warning("No date information available for multi-day stacked bar chart")
                bars = ax.bar(
                    np.arange(len(df)),
                    df['generation_kwh'],
                    width=0.6,
                    color=COLORS.get("generation", "#4CAF50"),
                    alpha=0.8
                )

                # Add data labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 5,
                        f'{height:.1f}',
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )

                # Set x-axis ticks
                ax.set_xticks(np.arange(len(df)))
                ax.set_xticklabels(df['tod_bin'])

        # Import the helper function to get plant display name
        from backend.data.data import get_plant_display_name

        # Get the display name for the plant
        plant_display_name = get_plant_display_name(plant_name)

        # Add labels and title
        if end_date is None or start_date == end_date:
            date_str = start_date.strftime('%Y-%m-%d')
            ax.set_title(f"ToD Generation for {plant_display_name} on {date_str}", fontsize=16, pad=20)
        else:
            date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            ax.set_title(f"ToD Generation for {plant_display_name} ({date_range})", fontsize=16, pad=20)

        ax.set_ylabel("Generation (kWh)", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)

        # Format y-axis with K for thousands
        ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))

        # Add grid for y-axis only
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        # Add legend for ToD categories
        ax.legend(loc='upper right', frameon=True, framealpha=0.9)

        # Add subtle watermark
        fig.text(0.99, 0.01, 'ToD Generation',
                fontsize=8, color='gray', ha='right', va='bottom', alpha=0.5)

        # Adjust layout
        plt.tight_layout()
        return fig

    except Exception as e:
        logger.error(f"Error creating ToD generation plot: {e}")
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}",
                ha='center', va='center', fontsize=12)
        return fig




def create_tod_consumption_plot(df, plant_name, start_date, end_date=None):
    """
    Create a stacked bar chart of consumption data with custom ToD bins
    based on the configuration settings.

    Args:
        df (DataFrame): ToD binned data with consumption
        plant_name (str): Name of the plant
        start_date (datetime): Start date
        end_date (datetime, optional): End date. If None, only start_date is used.

    Returns:
        Figure: Matplotlib figure object
    """
    try:
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 7))

        # Define ToD categories for stacking
        tod_categories = {
            0: '10 AM - 6 PM (Off-Peak)',
            1: '10 PM - 6 AM (Off-Peak)',
            2: '6 AM - 10 AM (Peak)',
            3: '6 PM - 10 PM (Peak)'
        }

        # Define colors for each ToD category - using distinct colors that complement the generation colors
        tod_colors = {
            '10 AM - 6 PM (Off-Peak)': '#00BCD4',  # Cyan for daytime off-peak
            '10 PM - 6 AM (Off-Peak)': '#673AB7',  # Deep Purple for nighttime off-peak
            '6 AM - 10 AM (Peak)': '#F44336',      # Red for morning peak
            '6 PM - 10 PM (Peak)': '#9C27B0'       # Purple for evening peak
        }

        is_single_day = end_date is None or start_date == end_date

        if is_single_day:
            # For single day view, create a stacked bar chart with ToD categories

            # Check if we have the right data structure
            if 'tod_bin' in df.columns and 'consumption_kwh' in df.columns:
                # Create a mapping from existing tod_bin to our categories
                # This assumes the tod_bin format matches what's in the data

                # Create a new dataframe with the data organized by our ToD categories
                stacked_data = {}

                # Initialize with zeros
                for date_val in df['date'].unique() if 'date' in df.columns else [start_date]:
                    stacked_data[date_val] = {cat: 0 for cat in tod_categories.values()}

                # Fill in the values from our data
                for _, row in df.iterrows():
                    tod_bin = row['tod_bin']
                    cons_kwh = row['consumption_kwh']
                    date_val = row['date'] if 'date' in df.columns else start_date

                    # Find the matching category
                    for _, cat in tod_categories.items():
                        if cat in tod_bin or tod_bin in cat:
                            stacked_data[date_val][cat] = cons_kwh
                            break

                # Convert to DataFrame for plotting
                plot_data = pd.DataFrame(stacked_data).T

                # Create the stacked bar chart
                bottom = np.zeros(len(plot_data))

                # Plot each category as a segment of the stacked bar
                for cat in tod_categories.values():
                    if cat in plot_data.columns:
                        ax.bar(
                            range(len(plot_data)),
                            plot_data[cat],
                            bottom=bottom,
                            label=cat,
                            color=tod_colors[cat],
                            alpha=0.8,
                            width=0.6
                        )
                        bottom += plot_data[cat].values

                # Add total value on top of the stacked bar
                for i, total in enumerate(plot_data.sum(axis=1)):
                    ax.text(
                        i,
                        total + 5,
                        f'{total:.1f}',
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        fontweight='bold'
                    )

                # Set x-axis labels
                if 'date' in df.columns:
                    ax.set_xticks(range(len(plot_data)))
                    ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in plot_data.index])
                else:
                    ax.set_xticks([0])
                    ax.set_xticklabels([start_date.strftime('%Y-%m-%d')])
            else:
                # Fallback to simple bar chart if data structure doesn't match
                logger.warning("Data structure doesn't match expected format for stacked bar chart")
                bars = ax.bar(
                    np.arange(len(df)),
                    df['consumption_kwh'],
                    width=0.6,
                    color=COLORS.get("consumption", "#F44336"),
                    alpha=0.8
                )

                # Add data labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 5,
                        f'{height:.1f}',
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )

                # Set x-axis ticks
                ax.set_xticks(np.arange(len(df)))
                ax.set_xticklabels(df['tod_bin'])
        else:
            # For multiple days view, create a stacked bar chart for each day

            # Check if we have date information
            if 'date' in df.columns:
                # Sort the dataframe by date
                df = df.sort_values('date')

                # Get unique dates
                unique_dates = df['date'].unique()
                num_dates = len(unique_dates)

                # Create a mapping from tod_bin to our categories
                tod_bin_to_category = {}
                for _, row in df.iterrows():
                    tod_bin = row['tod_bin']
                    for _, cat in tod_categories.items():
                        if cat in tod_bin or tod_bin in cat:
                            tod_bin_to_category[tod_bin] = cat
                            break

                # Create a new dataframe with data organized by date and ToD category
                plot_data = []

                # Process each date
                for date_val in unique_dates:
                    date_df = df[df['date'] == date_val]

                    # Initialize data for this date with zeros for all categories
                    date_data = {
                        'date': date_val,
                        'date_str': date_val.strftime('%Y-%m-%d')
                    }

                    # Initialize all categories with zero
                    for cat in tod_categories.values():
                        date_data[cat] = 0

                    # Fill in the values from our data
                    for _, row in date_df.iterrows():
                        tod_bin = row['tod_bin']
                        cons_kwh = row['consumption_kwh']

                        # Find the matching category
                        if tod_bin in tod_bin_to_category:
                            cat = tod_bin_to_category[tod_bin]
                            date_data[cat] = cons_kwh

                    plot_data.append(date_data)

                # Convert to DataFrame for plotting
                plot_df = pd.DataFrame(plot_data)

                # Set up x positions for the bars
                x = np.arange(num_dates)
                width = 0.6

                # Create the stacked bar chart - one stacked bar for each date
                bottom = np.zeros(num_dates)

                # Plot each category as a segment of the stacked bars
                for cat in tod_categories.values():
                    if cat in plot_df.columns:
                        ax.bar(
                            x,
                            plot_df[cat],
                            bottom=bottom,
                            label=cat,
                            color=tod_colors[cat],
                            alpha=0.8,
                            width=width
                        )
                        bottom += plot_df[cat].values

                # Add total value on top of each stacked bar
                for i, (_, row) in enumerate(plot_df.iterrows()):
                    total = sum(row[cat] for cat in tod_categories.values() if cat in row)
                    ax.text(
                        i,
                        total + 5,
                        f'{total:.1f}',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold'
                    )

                # Set x-axis labels
                ax.set_xticks(x)
                ax.set_xticklabels(plot_df['date_str'], rotation=45, ha='right')
            else:
                # Fallback to simple bar chart if no date information
                logger.warning("No date information available for multi-day stacked bar chart")
                bars = ax.bar(
                    np.arange(len(df)),
                    df['consumption_kwh'],
                    width=0.6,
                    color=COLORS.get("consumption", "#F44336"),
                    alpha=0.8
                )

                # Add data labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 5,
                        f'{height:.1f}',
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )

                # Set x-axis ticks
                ax.set_xticks(np.arange(len(df)))
                ax.set_xticklabels(df['tod_bin'])

        # Import the helper function to get plant display name
        from backend.data.data import get_plant_display_name

        # Get the display name for the plant
        plant_display_name = get_plant_display_name(plant_name)

        # Add labels and title
        if end_date is None or start_date == end_date:
            date_str = start_date.strftime('%Y-%m-%d')
            ax.set_title(f"ToD Consumption for {plant_display_name} on {date_str}", fontsize=16, pad=20)
        else:
            date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            ax.set_title(f"ToD Consumption for {plant_display_name} ({date_range})", fontsize=16, pad=20)

        ax.set_ylabel("Consumption (kWh)", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)

        # Format y-axis with K for thousands
        ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))

        # Add grid for y-axis only
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        # Add legend for ToD categories
        ax.legend(loc='upper right', frameon=True, framealpha=0.9)

        # Add subtle watermark
        fig.text(0.99, 0.01, 'ToD Consumption',
                fontsize=8, color='gray', ha='right', va='bottom', alpha=0.5)

        # Adjust layout
        plt.tight_layout()
        return fig

    except Exception as e:
        logger.error(f"Error creating ToD consumption plot: {e}")
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}",
                ha='center', va='center', fontsize=12)
        return fig






def create_daily_tod_binned_plot(df, plant_name, start_date, end_date):
    """
    Create a stacked area/bar plot for daily ToD Generation vs Consumption with custom time bins

    This function is designed for multi-day date ranges, showing generation and consumption
    patterns across the predefined time-of-day bins based on the configuration settings
    for each day in the selected range.

    Args:
        df (DataFrame): Daily ToD binned data with generation and consumption for multiple days
        plant_name (str): Name of the plant
        start_date (datetime): Start date of the data range
        end_date (datetime): End date of the data range

    Returns:
        Figure: Matplotlib figure object
    """
    # Import logger

    try:
        # Create the figure with single plot (removed summary table as per user preference)
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))

        # Log the dataframe structure for debugging
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame sample: {df.head(2).to_dict()}")

        # Import ToD configuration to get bin labels
        from backend.config.tod_config import get_tod_bin_labels

        # Get bin labels in the compact format for visualization
        tod_bins = get_tod_bin_labels("compact")

        # For multi-day data, we need to create a date-based structure
        # Check if we already have a date column
        if 'date' not in df.columns:
            # If no date column, we're working with aggregated data
            # Create a simple bar chart instead of a time series
            logger.info("No date column found, creating bar chart for aggregated data")

            # Ensure we have the tod_bin column
            if 'tod_bin' not in df.columns:
                logger.error("Required column 'tod_bin' not found in dataframe")
                raise ValueError("Required column 'tod_bin' not found in dataframe")

            # Create a bar chart for generation and consumption by ToD bin
            x = np.arange(len(df))
            width = 0.35

            # Sort by the defined bin order if possible
            if all(bin_name in tod_bins for bin_name in df['tod_bin']):
                df['bin_order'] = df['tod_bin'].apply(lambda x: tod_bins.index(x))
                df = df.sort_values('bin_order').drop('bin_order', axis=1)

            # Plot generation bars
            bars1 = ax1.bar(x - width/2, df['generation_kwh'], width, label='Generation',
                           color='#4CAF50', alpha=0.8)

            # Plot consumption bars
            bars2 = ax1.bar(x + width/2, df['consumption_kwh'], width, label='Consumption',
                           color='#F44336', alpha=0.8)

            # Add data labels
            def add_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax1.annotate(f'{height:.1f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)

            add_labels(bars1)
            add_labels(bars2)

            # Set x-axis labels with improved visibility
            ax1.set_xticks(x)
            # Make x-axis labels more readable
            labels = [bin_name.replace(' (Peak)', '\n(Peak)').replace(' (Off-Peak)', '\n(Off-Peak)')
                     for bin_name in df['tod_bin']]
            ax1.set_xticklabels(labels, fontsize=11, fontweight='bold')

            # Add grid, legend, and labels
            ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
            ax1.legend(loc='upper right', fontsize=11)
            ax1.set_ylabel('Energy (kWh)', fontsize=12, fontweight='bold')
            # Import the helper function to get plant display name
            from backend.data.data import get_plant_display_name

            # Get the display name for the plant
            plant_display_name = get_plant_display_name(plant_name)

            # ax1.set_title(f"ToD Generation vs Consumption for {plant_name}", fontsize=14, fontweight='bold')
            if end_date is None or start_date == end_date:
                date_str = start_date.strftime('%Y-%m-%d')
                ax1.set_title(f"ToD Generation vs Consumption for {plant_display_name} on {date_str}", fontsize=16, pad=20)
            else:
                date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                ax1.set_title(f"ToD Generation vs Consumption for {plant_display_name} ({date_range})", fontsize=16, pad=20)

            # Add more space at the bottom for x-axis labels
            plt.subplots_adjust(bottom=0.2)

            # Format y-axis with K for thousands
            ax1.yaxis.set_major_formatter(FuncFormatter(format_thousands))

            # Summary table removed as per user preference for ToD tab

        else:
            # We have a date column, create a time series visualization
            logger.info("Date column found, creating time series visualization")

            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'])

            # Create a pivot table with dates as rows and ToD bins as columns
            pivot_gen = df.pivot_table(
                index='date',
                columns='tod_bin',
                values='generation_kwh',
                aggfunc='sum'
            ).fillna(0)

            pivot_cons = df.pivot_table(
                index='date',
                columns='tod_bin',
                values='consumption_kwh',
                aggfunc='sum'
            ).fillna(0)

            # Ensure all expected ToD bins are present
            for bin_name in tod_bins:
                if bin_name not in pivot_gen.columns:
                    pivot_gen[bin_name] = 0
                if bin_name not in pivot_cons.columns:
                    pivot_cons[bin_name] = 0

            # Filter columns to include only the standard ToD bins
            pivot_gen = pivot_gen[tod_bins]
            pivot_cons = pivot_cons[tod_bins]

            # Create a colormap for the ToD bins
            colors_gen = ['#4CAF50', '#8BC34A', '#4CAF50', '#8BC34A']  # Green shades for generation
            colors_cons = ['#F44336', '#FF9800', '#F44336', '#FF9800']  # Red/orange for consumption

            # Plot stacked area chart for generation
            pivot_gen.plot(
                kind='area',
                stacked=True,
                ax=ax1,
                color=colors_gen,
                alpha=0.7,
                linewidth=0
            )

            # Plot stacked area chart for consumption as lines
            pivot_cons.plot(
                kind='line',
                ax=ax1,
                color=colors_cons,
                linewidth=2,
                marker='o',
                markersize=5
            )

            # Format x-axis dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            if (end_date - start_date).days > 30:
                ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            else:
                ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))

            # Format y-axis with K for thousands
            ax1.yaxis.set_major_formatter(FuncFormatter(format_thousands))

            # Add grid
            ax1.grid(True, linestyle='--', alpha=0.5)

            # Customize legend with better styling
            handles, labels = ax1.get_legend_handles_labels()
            # Reorder handles and labels to group generation and consumption
            new_handles = handles[:4] + handles[4:]
            new_labels = ['Gen: ' + label for label in labels[:4]] + ['Cons: ' + label for label in labels[4:]]
            ax1.legend(new_handles, new_labels, loc='upper left', bbox_to_anchor=(1.01, 1),
                      borderaxespad=0, fontsize=10, framealpha=0.9)

            # Improve x-axis labels
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10, fontweight='bold')

            # Set labels with better styling
            ax1.set_ylabel('Energy (kWh)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Date', fontsize=12, fontweight='bold')

            # Add more space at the bottom for x-axis labels
            plt.subplots_adjust(bottom=0.15)

            # Summary table removed as per user preference for ToD tab

        # Adjust layout for the plot (no table styling needed)

        # Adjust layout with improved spacing - don't use tight_layout() with subplots_adjust()
        # as they can conflict with each other
        plt.subplots_adjust(top=0.85, bottom=0.15, hspace=0.4, left=0.1, right=0.9)

        # Title is already set on the subplot (ax1), no need for a figure title

        return fig

    except Exception as e:
        logger.error(f"Error creating daily ToD binned plot: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}",
                ha='center', va='center', fontsize=12)
        return fig

def create_combined_wind_solar_plot(df, client_name, start_date, end_date):
    """
    Create a plot showing combined wind and solar generation for a client

    Args:
        df (DataFrame): Combined wind and solar generation data
        client_name (str): Name of the client
        start_date (datetime): Start date of the data
        end_date (datetime): End date of the data

    Returns:
        Figure: Matplotlib figure object
    """
    sns.set_theme(style="whitegrid")

    try:
        # Create figure with two subplots - line chart on left, pie chart on right
        fig = plt.figure(figsize=(16, 8))

        # Create a grid spec to control the width of the subplots
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])  # 2:1 ratio for line chart to pie chart

        # Create the two axes
        ax1 = fig.add_subplot(gs[0, 0])  # Line chart
        ax2 = fig.add_subplot(gs[0, 1])  # Pie chart

        # Print dataframe info for debugging
        logger.info(f"Combined wind and solar data shape: {df.shape}")
        logger.info(f"Combined wind and solar data columns: {df.columns.tolist()}")
        logger.info(f"Combined wind and solar data sample: {df.head(2).to_dict()}")

        # Create a copy to avoid modifying the original dataframe
        plot_df = df.copy()

        # Ensure all column names are lowercase
        plot_df.columns = [col.lower() for col in plot_df.columns]

        # Make sure we have the required columns
        if 'date' not in plot_df.columns:
            logger.error("Date column not found in dataframe")
            raise ValueError("Date column not found in dataframe")

        if 'source' not in plot_df.columns:
            logger.error("Source column not found in dataframe")
            raise ValueError("Source column not found in dataframe")

        if 'generation_kwh' not in plot_df.columns:
            logger.error("Generation_kwh column not found in dataframe")
            raise ValueError("Generation_kwh column not found in dataframe")

        # Make sure date column is datetime
        plot_df['date'] = pd.to_datetime(plot_df['date'])

        # Group by date and source to aggregate generation
        logger.info("Grouping data by date and source")
        grouped_df = plot_df.groupby([plot_df['date'].dt.date, 'source'])['generation_kwh'].sum().reset_index()

        # Convert date back to datetime for plotting
        grouped_df['date'] = pd.to_datetime(grouped_df['date'])

        # Pivot the data to get generation by date and source
        logger.info("Pivoting data")
        pivot_df = grouped_df.pivot(
            index='date',
            columns='source',
            values='generation_kwh'
        ).reset_index()

        # Fill NaN values with 0
        if 'Solar' in pivot_df.columns:
            pivot_df['Solar'] = pivot_df['Solar'].fillna(0)
        else:
            pivot_df['Solar'] = 0

        if 'Wind' in pivot_df.columns:
            pivot_df['Wind'] = pivot_df['Wind'].fillna(0)
        else:
            pivot_df['Wind'] = 0

        # Calculate total generation
        pivot_df['Total'] = pivot_df['Solar'] + pivot_df['Wind']

        # Calculate total generation by source for pie chart
        total_solar = pivot_df['Solar'].sum()
        total_wind = pivot_df['Wind'].sum()

        # ===== LINE CHART (LEFT SIDE) =====

        # Plot solar generation on the line chart
        ax1.plot(
            pivot_df['date'],
            pivot_df['Solar'],
            color=COLORS.get("secondary", "#FBBC05"),  # Yellow for solar
            marker='o',
            markersize=6,
            linewidth=2,
            label='Solar Generation'
        )

        # Plot wind generation on the line chart
        ax1.plot(
            pivot_df['date'],
            pivot_df['Wind'],
            color=COLORS.get("primary", "#4285F4"),  # Blue for wind
            marker='^',
            markersize=6,
            linewidth=2,
            label='Wind Generation'
        )

        # Plot total generation on the line chart
        ax1.plot(
            pivot_df['date'],
            pivot_df['Total'],
            color=COLORS.get("success", "#34A853"),  # Green for total
            marker='s',
            markersize=6,
            linewidth=3,
            label='Total Generation'
        )

        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        if (end_date - start_date).days > 30:
            ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        else:
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))

        # Format y-axis with K for thousands
        ax1.yaxis.set_major_formatter(FuncFormatter(format_thousands))

        # Add grid to line chart
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Axes labels for line chart
        ax1.set_xlabel("Date", fontsize=12)
        ax1.set_ylabel("Generation (kWh)", fontsize=12)

        # ===== PIE CHART (RIGHT SIDE) =====

        # Create pie chart data
        pie_data = [total_solar, total_wind]
        pie_labels = ['Solar', 'Wind']
        pie_colors = [COLORS.get("secondary", "#FBBC05"), COLORS.get("primary", "#4285F4")]

        # Calculate percentages for pie chart labels
        total_generation = total_solar + total_wind
        solar_percentage = (total_solar / total_generation * 100) if total_generation > 0 else 0
        wind_percentage = (total_wind / total_generation * 100) if total_generation > 0 else 0

        # Create pie chart labels with percentages
        pie_labels = [f'Solar: {solar_percentage:.1f}%', f'Wind: {wind_percentage:.1f}%']

        # Plot pie chart
        ax2.pie(
            pie_data,
            labels=pie_labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=pie_colors,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1},
            textprops={'fontsize': 12}
        )

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax2.axis('equal')

        # Add title to pie chart
        ax2.set_title('Distribution of Generation Sources', fontsize=14, pad=20)

        # Print column names for debugging
        logger.info(f"DataFrame columns for plant names: {df.columns.tolist()}")

        # Summary text box removed as requested

        # Add legend to the line chart
        ax1.legend(loc='upper right', frameon=True, framealpha=0.9)

        # Rotate x-axis labels on the line chart
        for label in ax1.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

        # Plant names text box removed as requested

        # Title for the entire figure
        date_range_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

        # Adjust layout - don't use tight_layout() with subplots_adjust()
        # as they can conflict with each other
        plt.subplots_adjust(top=0.85, bottom=0.15, hspace=0.3, left=0.1, right=0.9)

        # Add title after adjusting the layout
        fig.suptitle(f"Combined Wind and Solar Generation for {client_name} ({date_range_str})",
                    fontsize=16, y=0.98)

        return fig

    except Exception as e:
        logger.error(f"Error creating combined wind and solar plot: {e}")
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}",
                ha='center', va='center', fontsize=12)
        return fig


def create_power_cost_comparison_plot(cost_df, plant_name, start_date, end_date=None):
    """
    Create a comparison plot showing grid cost vs actual cost.

    Args:
        cost_df (DataFrame): DataFrame with cost metrics
        plant_name (str): Name of the plant
        start_date (datetime): Start date
        end_date (datetime, optional): End date. If None, only start_date is used.

    Returns:
        Figure: Matplotlib figure object
    """
    try:
        if cost_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "No data available for cost analysis",
                    ha='center', va='center', fontsize=12)
            return fig

        # Validate required columns
        required_columns = ['grid_cost', 'actual_cost']
        missing_columns = [col for col in required_columns if col not in cost_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns for cost comparison plot: {missing_columns}")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, f"Missing data columns: {', '.join(missing_columns)}",
                   ha='center', va='center', fontsize=12)
            ax.set_title("Power Cost Comparison - Data Error")
            return fig

        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Determine if single day or date range
        is_single_day = end_date is None or start_date == end_date

        if is_single_day:
            # For single day, use time column or create index
            if 'time' in cost_df.columns:
                x_data = cost_df['time']
                x_label = "Time"
            else:
                x_data = range(len(cost_df))
                x_label = "Time Period"
            title_date = start_date.strftime('%Y-%m-%d')
        else:
            # For date range, use date column
            if 'date' in cost_df.columns:
                x_data = cost_df['date']
                x_label = "Date"
            else:
                x_data = range(len(cost_df))
                x_label = "Day"
            title_date = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

        # Create bar chart comparing grid cost vs actual cost
        width = 0.35
        x_pos = range(len(cost_df))

        bars1 = ax.bar([x - width/2 for x in x_pos], cost_df['grid_cost'],
                      width, label='Grid Cost (Without Solar/Wind)',
                      color=COLORS.get("danger", "#EA4335"), alpha=0.8)

        bars2 = ax.bar([x + width/2 for x in x_pos], cost_df['actual_cost'],
                      width, label='Actual Cost (With Solar/Wind)',
                      color=COLORS.get("success", "#34A853"), alpha=0.8)

        # Value labels removed as requested

        # Customize the plot
        from backend.data.data import get_plant_display_name
        plant_display_name = get_plant_display_name(plant_name)

        ax.set_title(f"Power Cost Comparison - {plant_display_name}\n{title_date}", fontsize=14, pad=20)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Cost (₹)", fontsize=12)
        ax.legend()

        # Format x-axis
        if is_single_day:
            # For single day data
            step = max(1, len(cost_df) // 8)
            ax.set_xticks(range(0, len(cost_df), step))

            if 'time' in cost_df.columns:
                # Try to format time column
                try:
                    labels = []
                    for i in range(0, len(cost_df), step):
                        time_val = cost_df.iloc[i]['time']
                        if hasattr(time_val, 'strftime'):
                            labels.append(time_val.strftime('%H:%M'))
                        else:
                            labels.append(f"T{i}")
                    ax.set_xticklabels(labels, rotation=45)
                except:
                    # Fallback to simple numbering
                    ax.set_xticklabels([f"T{i}" for i in range(0, len(cost_df), step)], rotation=45)
            else:
                # Use simple time period labels
                ax.set_xticklabels([f"Period {i}" for i in range(0, len(cost_df), step)], rotation=45)
        else:
            # For date range data
            step = max(1, len(cost_df) // 10)
            ax.set_xticks(range(0, len(cost_df), step))

            if 'date' in cost_df.columns:
                try:
                    labels = []
                    for i in range(0, len(cost_df), step):
                        date_val = cost_df.iloc[i]['date']
                        if hasattr(date_val, 'strftime'):
                            labels.append(date_val.strftime('%m/%d'))
                        elif hasattr(date_val, 'year'):  # Handle date objects
                            labels.append(f"{date_val.month:02d}/{date_val.day:02d}")
                        else:
                            labels.append(f"Day {i+1}")
                    ax.set_xticklabels(labels, rotation=45)
                except Exception as e:
                    logger.warning(f"Error formatting date labels: {e}")
                    # Fallback to simple numbering
                    ax.set_xticklabels([f"Day {i+1}" for i in range(0, len(cost_df), step)], rotation=45)
            else:
                # Use simple day labels
                ax.set_xticklabels([f"Day {i+1}" for i in range(0, len(cost_df), step)], rotation=45)

        # Add grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        return fig

    except Exception as e:
        logger.error(f"Error creating power cost comparison plot: {e}")
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}",
                ha='center', va='center', fontsize=12)
        return fig


def create_power_savings_plot(cost_df, plant_name, start_date, end_date=None):
    """
    Create a plot showing power savings over time.

    Args:
        cost_df (DataFrame): DataFrame with cost metrics
        plant_name (str): Name of the plant
        start_date (datetime): Start date
        end_date (datetime, optional): End date. If None, only start_date is used.

    Returns:
        Figure: Matplotlib figure object
    """
    try:
        if cost_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "No data available for savings analysis",
                    ha='center', va='center', fontsize=12)
            return fig

        # Validate required columns
        if 'savings' not in cost_df.columns:
            logger.error(f"Missing 'savings' column for power savings plot. Available columns: {cost_df.columns.tolist()}")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "Missing savings data column",
                   ha='center', va='center', fontsize=12)
            ax.set_title("Power Savings - Data Error")
            return fig

        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Determine if single day or date range
        is_single_day = end_date is None or start_date == end_date

        if is_single_day:
            # For single day, use line plot
            if 'time' in cost_df.columns:
                # Try to use time data for x-axis
                try:
                    time_data = pd.to_datetime(cost_df['time'])
                    ax.plot(range(len(cost_df)), cost_df['savings'], marker='o', linewidth=2,
                           color=COLORS.get("success", "#34A853"), markersize=4)

                    # Format x-axis with time labels
                    step = max(1, len(cost_df) // 8)
                    ax.set_xticks(range(0, len(cost_df), step))
                    labels = []
                    for i in range(0, len(cost_df), step):
                        time_val = time_data.iloc[i]
                        if hasattr(time_val, 'strftime'):
                            labels.append(time_val.strftime('%H:%M'))
                        else:
                            labels.append(f"T{i}")
                    ax.set_xticklabels(labels, rotation=45)
                except:
                    # Fallback to simple plot
                    ax.plot(range(len(cost_df)), cost_df['savings'], marker='o', linewidth=2,
                           color=COLORS.get("success", "#34A853"), markersize=4)
            else:
                # Use simple index-based plot
                ax.plot(range(len(cost_df)), cost_df['savings'], marker='o', linewidth=2,
                       color=COLORS.get("success", "#34A853"), markersize=4)

            x_label = "Time"
            title_date = start_date.strftime('%Y-%m-%d')
        else:
            # For date range, use bar plot
            bars = ax.bar(range(len(cost_df)), cost_df['savings'],
                         color=COLORS.get("success", "#34A853"), alpha=0.8)

            # Add value labels on bars
            for bar in enumerate(bars):
                height = bar[1].get_height()
                ax.text(bar[1].get_x() + bar[1].get_width()/2., height + height*0.01,
                       f'₹{height:.0f}', ha='center', va='bottom', fontsize=9)

            x_label = "Date"
            title_date = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

            # Format x-axis for dates
            step = max(1, len(cost_df) // 10)
            ax.set_xticks(range(0, len(cost_df), step))
            if 'date' in cost_df.columns:
                try:
                    labels = []
                    for i in range(0, len(cost_df), step):
                        date_val = cost_df.iloc[i]['date']
                        if hasattr(date_val, 'strftime'):
                            labels.append(date_val.strftime('%m/%d'))
                        elif hasattr(date_val, 'year'):  # Handle date objects
                            labels.append(f"{date_val.month:02d}/{date_val.day:02d}")
                        else:
                            labels.append(f"Day {i+1}")
                    ax.set_xticklabels(labels, rotation=45)
                except Exception as e:
                    logger.warning(f"Error formatting date labels in savings plot: {e}")
                    ax.set_xticklabels([f"Day {i+1}" for i in range(0, len(cost_df), step)], rotation=45)
            else:
                ax.set_xticklabels([f"Day {i+1}" for i in range(0, len(cost_df), step)], rotation=45)

        # Customize the plot
        from backend.data.data import get_plant_display_name
        plant_display_name = get_plant_display_name(plant_name)

        ax.set_title(f"Power Cost Savings - {plant_display_name}\n{title_date}", fontsize=14, pad=20)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Savings (₹)", fontsize=12)

        # Add grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        # Add zero line if there are negative savings
        if cost_df['savings'].min() < 0:
            ax.axhline(y=0, color='red', linestyle='-', alpha=0.5)

        plt.tight_layout()
        return fig

    except Exception as e:
        logger.error(f"Error creating power savings plot: {e}")
        logger.error(traceback.format_exc())
        # Return a simple error figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}",
                ha='center', va='center', fontsize=12)
        return fig


def create_banking_plot(df, plant_name, banking_type="daily", tod_based=False):
    """
    Create a plot for banking data

    Args:
        df (DataFrame): Banking data
        plant_name (str): Name of the plant
        banking_type (str): Type of banking data (daily, monthly, yearly)
        tod_based (bool): Whether the data is ToD-based

    Returns:
        Figure: Matplotlib figure object
    """
    try:
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Set the title based on the banking type and ToD setting
        tod_text = "ToD-Based" if tod_based else "Non-ToD"
        title = f"{banking_type.capitalize()} Banking ({tod_text}) - {plant_name}"
        ax.set_title(title, fontsize=16, pad=20)

        # Determine the columns to plot based on the banking type and ToD setting
        if tod_based:
            # For ToD-based banking, we have different columns
            if 'origin_slot_name' in df.columns:
                # Create a grouped bar chart for peak and off-peak
                peak_df = df[df['origin_slot_name'] == 'peak']
                offpeak_df = df[df['origin_slot_name'] == 'offpeak']

                # Determine x-axis based on banking type
                if banking_type == "daily":
                    x_col = 'Date'
                elif banking_type == "monthly":
                    x_col = 'Month'
                else:  # yearly
                    x_col = 'Year'

                # Plot peak data
                if not peak_df.empty:
                    sns.barplot(
                        data=peak_df,
                        x=x_col,
                        y='Surplus Generation(After Settlement)',
                        color=COLORS.get("primary", "#1E88E5"),
                        alpha=0.8,
                        label="Peak Surplus Generation",
                        ax=ax
                    )

                    # Plot grid consumption as negative values
                    sns.barplot(
                        data=peak_df,
                        x=x_col,
                        y='Grid Consumption(After Settlement)',
                        color=COLORS.get("consumption", "#00897B"),
                        alpha=0.8,
                        label="Peak Grid Consumption",
                        ax=ax
                    )

                # Plot off-peak data
                if not offpeak_df.empty:
                    sns.barplot(
                        data=offpeak_df,
                        x=x_col,
                        y='Surplus Generation(After Settlement)',
                        color=COLORS.get("secondary", "#5E35B1"),
                        alpha=0.8,
                        label="Off-Peak Surplus Generation",
                        ax=ax
                    )

                    # Plot grid consumption as negative values
                    sns.barplot(
                        data=offpeak_df,
                        x=x_col,
                        y='Grid Consumption(After Settlement)',
                        color=COLORS.get("tertiary", "#00ACC1"),
                        alpha=0.8,
                        label="Off-Peak Grid Consumption",
                        ax=ax
                    )
            else:
                # Fallback if the expected columns aren't found
                ax.text(0.5, 0.5, "No ToD data available", ha='center', va='center', fontsize=14)
        else:
            # For non-ToD banking
            if banking_type == "daily":
                # For daily banking, plot Surplus Generation and Grid Consumption
                if 'Surplus Generation' in df.columns and 'Grid Consumption' in df.columns:
                    # Plot surplus generation
                    sns.barplot(
                        data=df,
                        x='Date',
                        y='Surplus Generation',
                        color=COLORS.get("primary", "#1E88E5"),
                        alpha=0.8,
                        label="Surplus Generation",
                        ax=ax
                    )

                    # Plot grid consumption
                    sns.barplot(
                        data=df,
                        x='Date',
                        y='Grid Consumption',
                        color=COLORS.get("consumption", "#00897B"),
                        alpha=0.8,
                        label="Grid Consumption",
                        ax=ax
                    )
                else:
                    ax.text(0.5, 0.5, "No daily banking data available", ha='center', va='center', fontsize=14)

            elif banking_type == "monthly":
                # For monthly banking, plot Surplus Generation and Grid Consumption
                if 'Surplus Generation' in df.columns and 'Grid Consumption' in df.columns:
                    # Plot surplus generation
                    sns.barplot(
                        data=df,
                        x='Month',
                        y='Surplus Generation',
                        color=COLORS.get("primary", "#1E88E5"),
                        alpha=0.8,
                        label="Surplus Generation",
                        ax=ax
                    )

                    # Plot grid consumption
                    sns.barplot(
                        data=df,
                        x='Month',
                        y='Grid Consumption',
                        color=COLORS.get("consumption", "#00897B"),
                        alpha=0.8,
                        label="Grid Consumption",
                        ax=ax
                    )
                else:
                    ax.text(0.5, 0.5, "No monthly banking data available", ha='center', va='center', fontsize=14)

            elif banking_type == "yearly":
                # For yearly banking, plot Yearly Surplus and Yearly Deficit
                if 'Yearly Surplus' in df.columns and 'Yearly Deficit' in df.columns:
                    # Plot yearly surplus
                    sns.barplot(
                        data=df,
                        x='Year',
                        y='Yearly Surplus',
                        color=COLORS.get("primary", "#1E88E5"),
                        alpha=0.8,
                        label="Yearly Surplus",
                        ax=ax
                    )

                    # Plot yearly deficit
                    sns.barplot(
                        data=df,
                        x='Year',
                        y='Yearly Deficit',
                        color=COLORS.get("consumption", "#00897B"),
                        alpha=0.8,
                        label="Yearly Deficit",
                        ax=ax
                    )
                else:
                    ax.text(0.5, 0.5, "No yearly banking data available", ha='center', va='center', fontsize=14)

        # Add legend
        ax.legend()

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.7)

        # Rotate x-axis labels if needed
        plt.xticks(rotation=45)

        # Adjust layout manually instead of using tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)

        return fig
    except Exception as e:
        logger.error(f"Error creating banking plot: {e}")
        # Create an empty figure with error message
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Error creating banking plot: {str(e)}", ha='center', va='center', fontsize=14)
        return fig
