def get_unique_values(df, column_name):
    """
    Get unique values for a specified column in the dataframe.
    
    Parameters:
    df (pandas.DataFrame): The dataframe to analyze
    column_name (str): The name of the column to get unique values for
    
    Returns:
    list: List of unique values in the column, or None if column doesn't exist
    """
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in dataframe.")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    unique_vals = df[column_name].unique().tolist()
    print(f"Unique values in '{column_name}' column ({len(unique_vals)} total):")
    print(unique_vals)
    
    return unique_vals


