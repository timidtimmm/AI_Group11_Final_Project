import pandas as pd

def main():
    #load the dataset
    df = pd.read_csv("/home/weichen/AI/project/AI_Group11_Final_Project/total_dataset_final.csv")

    #check for correlation with 'volumn'
    if 'weekday' in df.columns:
        df['weekday'] = df['weekday'].map({'Y': 1, 'N': 0})
    numeric_df = df.select_dtypes(include='number')
    correlation = numeric_df.corr()['volumn'].sort_values(ascending=False)
    #print the correlation values
    print("Correlation with 'volumn':")
    print(correlation)
    #correlation data
    correlation.to_csv("correlation_with_volumn.csv", header=True)
    
if __name__ == "__main__":
    main()
