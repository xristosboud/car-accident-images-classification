import pandas as pd
import numpy as np

def load_data(file_path):
    """Load the data from an Excel file."""
    try:
        df = pd.read_excel(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

def preprocess_data(df):
    """Preprocess the DataFrame to extract necessary information and create dummy variables."""
    # Split the 'path' column into components
    df[['trash', 'claim_id', 'img_name']] = df['path'].str.split('\\', expand=True)
    
    # Drop unnecessary columns
    df.drop(columns=["path", "trash"], inplace=True)

    # Create dummy variables for the 'orientation' column
    df = pd.get_dummies(df, columns=['orientation'])
    
    return df

def aggregate_data(df):
    """Aggregate the data to get counts and percentages for each claim."""
    df_agg = df.groupby('claim_id').agg(
        N_back=pd.NamedAgg(column='orientation_back', aggfunc='sum'),
        N_front=pd.NamedAgg(column='orientation_front', aggfunc='sum'),
        N_side=pd.NamedAgg(column='orientation_side', aggfunc='sum')
    )

    # Calculate total counts and percentages
    df_agg["N_Total"] = df_agg[['N_back', 'N_front', 'N_side']].sum(axis=1)
    
    # Avoid division by zero in percentage calculations
    df_agg['%_back'] = 100 * df_agg['N_back'] / df_agg['N_Total'].replace(0, np.nan)
    df_agg['%_front'] = 100 * df_agg['N_front'] / df_agg['N_Total'].replace(0, np.nan)
    df_agg['%_side'] = 100 * df_agg['N_side'] / df_agg['N_Total'].replace(0, np.nan)

    return df_agg

def save_results(df_agg, output_file):
    """Save the aggregated results to an Excel file."""
    df_agg.reset_index().to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    """Main function to run the data analysis."""
    input_file = 'outputs/img_orient.xlsx'
    output_file = "outputs/img_stats.xlsx"

    # Load the data
    df = load_data(input_file)
    if df is None:
        return  

    # Preprocess the data
    df_processed = preprocess_data(df)

    # Aggregate the data
    df_aggregated = aggregate_data(df_processed)

    # Save the results
    save_results(df_aggregated, output_file)

if __name__ == "__main__":
    main()
