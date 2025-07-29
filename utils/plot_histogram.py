"""
Histogram plotting utilities using Seaborn for data visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_column_histogram(df, column_name, bins='auto', figsize=(10, 6), 
                         title=None, color='skyblue', kde=True, stat='count'):
    """
    Create a histogram for a specified column in a pandas DataFrame using Seaborn.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    column_name : str
        Name of the column to plot
    bins : int or str, default='auto'
        Number of bins or binning strategy
    figsize : tuple, default=(10, 6)
        Figure size (width, height)
    title : str, optional
        Custom title for the plot. If None, uses column name
    color : str, default='skyblue'
        Color of the histogram bars
    kde : bool, default=True
        Whether to overlay a kernel density estimate
    stat : str, default='count'
        Statistic to compute ('count', 'frequency', 'density', 'probability')
    
    Returns:
    --------
    None : Displays the histogram plot
    """
    
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in DataFrame.")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Check if column has numeric data
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        print(f"Warning: Column '{column_name}' is not numeric. Use plot_categorical_histogram() instead.")
        return
    
    # Remove missing values for the plot
    data_clean = df[column_name].dropna()
    
    if len(data_clean) == 0:
        print(f"Error: No valid data found in column '{column_name}' after removing NaN values.")
        return
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Create histogram with seaborn
    sns.histplot(data=data_clean, bins=bins, color=color, kde=kde, stat=stat, alpha=0.7)
    
    # Customize the plot
    if title is None:
        title = f'Distribution of {column_name.replace("_", " ").title()}'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(column_name.replace("_", " ").title(), fontsize=12)
    plt.ylabel(stat.title(), fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Count: {len(data_clean)}\n'
    stats_text += f'Mean: {data_clean.mean():.2f}\n'
    stats_text += f'Median: {data_clean.median():.2f}\n'
    stats_text += f'Std: {data_clean.std():.2f}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics for '{column_name}':")
    print(f"Count: {len(data_clean):,}")
    print(f"Mean: {data_clean.mean():.2f}")
    print(f"Median: {data_clean.median():.2f}")
    print(f"Standard Deviation: {data_clean.std():.2f}")
    print(f"Min: {data_clean.min():.2f}")
    print(f"Max: {data_clean.max():.2f}")
    if len(data_clean) != len(df[column_name]):
        print(f"Missing values: {len(df[column_name]) - len(data_clean)}")


def plot_categorical_histogram(df, column_name, hue=None, order=None, hue_order=None, 
                              figsize=(10, 6), title=None, palette='Set2', 
                              show_percentages=True):
    """
    Create a histogram for categorical data using Seaborn countplot.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    column_name : str
        Name of the categorical column to plot
    hue : str, optional
        Column name for grouping bars by different categories
    order : list, optional
        Order of categories on x-axis
    hue_order : list, optional
        Order of hue categories
    figsize : tuple, default=(10, 6)
        Figure size (width, height)
    title : str, optional
        Custom title for the plot. If None, uses column name
    palette : str, default='Set2'
        Color palette for the bars
    show_percentages : bool, default=True
        Whether to show percentages on top of bars
    
    Returns:
    --------
    None : Displays the histogram plot
    """
    
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in DataFrame.")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Validate hue column if provided
    if hue and hue not in df.columns:
        print(f"Error: Hue column '{hue}' not found in DataFrame.")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Remove missing values from main column and hue column if specified
    if hue:
        data_clean = df[[column_name, hue]].dropna()
        if len(data_clean) == 0:
            print(f"Error: No valid data found after removing NaN values from '{column_name}' and '{hue}'.")
            return
    else:
        data_clean = df[column_name].dropna()
        if len(data_clean) == 0:
            print(f"Error: No valid data found in column '{column_name}' after removing NaN values.")
            return
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Create count plot with hue support - use the original df but ensure order is respected
    ax = sns.countplot(data=df, x=column_name, hue=hue, order=order, 
    hue_order=hue_order, palette=palette)
    
    # If order is specified, ensure all categories are present even with zero counts
    if order and not hue:
        # For single variable plots, make sure all ordered categories appear
        current_labels = [t.get_text() for t in ax.get_xticklabels()]
        if set(current_labels) != set(order):
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order)
    
    # Customize the plot
    if title is None:
        if hue:
            title = f'Distribution of {column_name.replace("_", " ").title()} by {hue.replace("_", " ").title()}'
        else:
            title = f'Distribution of {column_name.replace("_", " ").title()}'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(column_name.replace("_", " ").title(), fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add percentages on top of bars
    if show_percentages:
        if hue:
            # For grouped bars, calculate percentages within each group
            total = len(data_clean)
            for p in ax.patches:
                if p.get_height() > 0:  # Only add labels to bars with data
                    percentage = f'{100 * p.get_height() / total:.1f}%'
                    ax.annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()),
                               ha='center', va='bottom', fontsize=9)
        else:
            # For single bars, use original logic
            total = len(data_clean)
            for p in ax.patches:
                percentage = f'{100 * p.get_height() / total:.1f}%'
                ax.annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()),
                           ha='center', va='bottom', fontsize=10)
    
    # Add legend if hue is used
    if hue:
        plt.legend(title=hue.replace("_", " ").title(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Print value counts
    if hue:
        # Cross-tabulation for grouped data
        crosstab = pd.crosstab(df[column_name], df[hue], margins=True)
        print(f"\nCross-tabulation of '{column_name}' by '{hue}':")
        print(crosstab)
        
        # Print percentages by group
        print(f"\nPercentages by '{hue}' groups:")
        crosstab_pct = pd.crosstab(df[column_name], df[hue], normalize='columns') * 100
        print(crosstab_pct.round(1))
    else:
        # Single column value counts
        if isinstance(data_clean, pd.Series):
            value_counts = data_clean.value_counts()
        else:
            value_counts = data_clean[column_name].value_counts()
            
        if order:
            value_counts = value_counts.reindex(order, fill_value=0)
        
        print(f"\nValue counts for '{column_name}':")
        total_count = len(data_clean)
        for value, count in value_counts.items():
            percentage = (count / total_count) * 100
            print(f"  {value}: {count:,} ({percentage:.1f}%)")
    
    # Report missing values
    original_len = len(df)
    clean_len = len(data_clean)
    if clean_len != original_len:
        print(f"\nMissing values removed: {original_len - clean_len}")


def plot_multiple_histograms(df, columns, ncols=2, figsize=(15, 10), 
                           palette='Set1', suptitle=None):
    """
    Create multiple histograms in a subplot grid.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    columns : list
        List of column names to plot
    ncols : int, default=2
        Number of columns in the subplot grid
    figsize : tuple, default=(15, 10)
        Figure size (width, height)
    palette : str, default='Set1'
        Color palette for the plots
    suptitle : str, optional
        Overall title for the subplot grid
    
    Returns:
    --------
    None : Displays the histogram plots
    """
    
    # Calculate number of rows needed
    nrows = (len(columns) + ncols - 1) // ncols
    
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    # Get colors from palette
    colors = sns.color_palette(palette, len(columns))
    
    for idx, column in enumerate(columns):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        if column not in df.columns:
            ax.text(0.5, 0.5, f"Column '{column}'\nnot found", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Error: {column}")
            continue
        
        # Check if numeric or categorical
        if pd.api.types.is_numeric_dtype(df[column]):
            # Numeric histogram
            data_clean = df[column].dropna()
            if len(data_clean) > 0:
                sns.histplot(data=data_clean, ax=ax, color=colors[idx], alpha=0.7)
        else:
            # Categorical histogram
            sns.countplot(data=df, x=column, ax=ax, color=colors[idx])
            ax.tick_params(axis='x', rotation=45)
        
        ax.set_title(column.replace("_", " ").title())
        ax.grid(axis='y', alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(columns), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'numeric_col': np.random.normal(50, 15, 1000),
        'categorical_col': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'temperature': np.random.choice([30, 55, 80], 1000)
    })
    
    print("Sample usage of histogram functions:")
    print("1. plot_column_histogram(df, 'numeric_col')")
    print("2. plot_categorical_histogram(df, 'categorical_col')")
    print("3. plot_multiple_histograms(df, ['numeric_col', 'categorical_col'])")
