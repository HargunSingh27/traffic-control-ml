# Import the pandas library, the cornerstone of our data analysis toolkit.
# We use the conventional alias 'pd' for brevity and readability.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define a constant for the path to our dataset file.
# This makes the script cleaner and easier to modify if the file name changes.
CSV_PATH = "features.csv"

# It's a good practice to wrap file operations in a try-except block.
# This handles potential errors gracefully, such as if the file is not found.
try:
    # Use the pd.read_csv() function to load the dataset.
    # The function reads the file and returns a pandas DataFrame, which we
    # store in a variable, conventionally named 'df' or a more descriptive name.
    features_df = pd.read_csv(CSV_PATH)

    print("DataFrame loaded successfully!")
      # Compute the correlation matrix for all numerical features in the DataFrame.
    print("\\n--- Computing Correlation Matrix ---")
    correlation_matrix = features_df.corr()

    print("\n--- Generating Heatmap of Feature Correlations ---")
    
    # Set up the matplotlib figure
    # A larger figure size is needed to make the heatmap readable.
    plt.figure(figsize=(18, 15))

    # Create the heatmap using Seaborn
    # cmap='coolwarm': This is a "diverging" colormap, ideal for correlation matrices.
    #   Positive correlations will be warm (red), negative will be cool (blue),
    #   and correlations near zero will be neutral.
    # annot=False: For a large matrix like this, annotating each cell with its
    #   value would make the plot unreadable. We leave it false.
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)

    # Set the plot title
    plt.title('Correlation Matrix of Music Features', fontsize=20)

    # Display the plot
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure you have run the 'feature_extractor.py' script first to generate the dataset.")
except Exception as e:
    print(f"An error occurred while loading the DataFrame: {e}")
