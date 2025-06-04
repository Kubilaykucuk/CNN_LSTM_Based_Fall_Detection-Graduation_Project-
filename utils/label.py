import pandas as pd
import os

def auto_label(data):
    """
    Labels radar data with the following scheme:
    0: Nothing
    1: Falling
    2: Walking

    Parameters:
    - data (DataFrame): must include ['X_m', 'Y_m', 'Range', 'RadialVelocity'] columns

    Returns:
    - DataFrame with a new 'Label' column
    """

    # Copy the input to avoid modifying original
    df = data.copy()

    # Initialize all labels to 0 (nothing)
    df['Posture'] = 0

    # Thresholds
    Z_DIFF_THRESHOLD = 0.1
    FALLING_VZ_THRESHOLD = 0.5
    MIN_INITIAL_Z = 1.0
    XY_MOVEMENT_THRESHOLD = 3

    # Calculate Z difference per object
    df['Prev_Z'] = df.groupby('ObjectID')['Z(m)'].shift(1)
    df['Z_diff'] = df['Z(m)'] - df['Prev_Z']
    df['X_diff'] = df.groupby('ObjectID')['X(m)'].diff().abs()
    df['Y_diff'] = df.groupby('ObjectID')['Y(m)'].diff().abs()


    # Fall condition: large Z change or fast vertical motion, per object
    falling_condition = (
        (df['Prev_Z'] > MIN_INITIAL_Z) & (
            (df['Z_diff'].abs() > Z_DIFF_THRESHOLD) |
            (df['Vz(m/s)'].abs() > FALLING_VZ_THRESHOLD)
        )
    )
    df.loc[falling_condition, 'Posture'] = 1

    # Walking condition: high radial velocity, not falling
    walking_condition = (
        (df['Posture'] != 1) &
        ((df['X_diff'] > XY_MOVEMENT_THRESHOLD) | (df['Y_diff'] > XY_MOVEMENT_THRESHOLD))
    )
    df.loc[walking_condition, 'Posture'] = 2

    return df

# Set source and target directories
source_folder = "test_dataset"
output_folder = "dataset"

# Make sure 'dataset' folder exists
os.makedirs("dataset", exist_ok=True)

# Loop through CSV files
for file in os.listdir(source_folder):
    if file.endswith(".csv") and "label" not in file.lower():
        file_path = os.path.join(source_folder, file)
        try:
            # Read the data
            df = pd.read_csv(file_path)

            # Apply labeling
            labeled_df = auto_label(df)

            # Save labeled file into 'dataset' folder
            new_filename = f"labeled_{file}"
            labeled_df.to_csv(os.path.join(output_folder, new_filename), index=False)

            print(f"Labeled and saved: {new_filename}")
        except Exception as e:
            print(f"Error processing {file}: {e}")