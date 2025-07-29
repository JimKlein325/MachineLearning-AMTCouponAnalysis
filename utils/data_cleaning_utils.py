# Safe column dropping
def safe_drop_columns(df, columns_to_drop):
    """Safely drop columns that exist in the DataFrame"""
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    if existing_columns:
        print(f"Dropping columns: {existing_columns}")
        return df.drop(columns=existing_columns)
    else:
        print("No columns to drop")
        return df
    
    



def analyze_missing_data(df):
    """
    Count missing data items and calculate percentage for the overall dataframe.
    
    Parameters:
    df (pandas.DataFrame): The dataframe to analyze
    
    Returns:
    dict: Dictionary containing missing data statistics
    """
    total_cells = df.size  # Total number of cells in the dataframe
    missing_cells = df.isnull().sum().sum()  # Total number of missing cells
    missing_percentage = (missing_cells / total_cells) * 100
    
    print("=== Missing Data Analysis ===")
    print(f"Total cells in dataframe: {total_cells:,}")
    print(f"Missing cells: {missing_cells:,}")
    print(f"Percentage of missing data: {missing_percentage:.2f}%")
    print(f"Complete data percentage: {100 - missing_percentage:.2f}%")
    
    # Also show per-column breakdown
    print("\n--- Missing Data by Column ---")
    missing_by_column = df.isnull().sum()
    missing_by_column_pct = (missing_by_column / len(df)) * 100
    
    for col in df.columns:
        if missing_by_column[col] > 0:
            print(f"{col}: {missing_by_column[col]} ({missing_by_column_pct[col]:.1f}%)")
    
    return {
        'total_cells': total_cells,
        'missing_cells': missing_cells,
        'missing_percentage': missing_percentage,
        'complete_percentage': 100 - missing_percentage,
        'missing_by_column': missing_by_column.to_dict(),
        'missing_by_column_pct': missing_by_column_pct.to_dict()
    }


