import pandas as pd
import cupy as np

# Read the dataset CSV file, including only the specified columns
dtype_dict = {
    'problem_id': str,
    'skill_id': str,
    'user_id': str,
    'order_id': int,
    'correct': int,
    'hint_count': int
}
columns_to_read = [
    'problem_id',
    'skill_id',
    'user_id',
    'order_id',
    'correct',
    'hint_count'
]
df = pd.read_csv(
    'skill_builder_data_corrected.csv',
    encoding='latin-1',
    usecols=columns_to_read,
    dtype=dtype_dict
)

# Select the relevant columns representing problem items
problem_col = 'problem_id'
# and KCs
kc_col = 'skill_id'

# Get the 20 most frequent KCs only
kc_counts = df[kc_col].value_counts()
top_kcs = kc_counts.head(20).index

# Filter the dataset to keep only the top KCs
df_filtered = df[df[kc_col].isin(top_kcs)].copy()

# Filter the dataset to keep only questions with 0 or 1 hint
df = df[df["hint_count"] <= 1]

# Sort the exercising trajectories and divide into time windows
df_filtered.sort_values(by=['user_id', 'order_id'], inplace=True)
df_filtered['time_window'] = df_filtered.groupby('user_id').cumcount() // 6 + 1

# Split the dataset into training and testing based on time windows
train_data = df_filtered[df_filtered['time_window'] < 7].copy()
test_data = df_filtered[df_filtered['time_window'] == 7].copy()

# Create the Q-matrix using the filtered dataset
problems = df_filtered[problem_col].unique()
kcs = df_filtered[kc_col].unique()

# Create a dictionary to map original IDs to new IDs
problem_id_map = {problem_id: new_id for new_id, problem_id in enumerate(problems)}
kc_id_map = {kc_id: new_id for new_id, kc_id in enumerate(kcs)}
user_id_map = {user_id: new_id for new_id, user_id in enumerate(df['user_id'].unique())}

# Update id columns with new IDs
train_data['problem_id'] = train_data['problem_id'].map(problem_id_map)
train_data['skill_id'] = train_data['skill_id'].map(kc_id_map)
train_data['user_id'] = train_data['user_id'].map(user_id_map)

test_data['problem_id'] = test_data['problem_id'].map(problem_id_map)
test_data['skill_id'] = test_data['skill_id'].map(kc_id_map)
test_data['user_id'] = test_data['user_id'].map(user_id_map)

q_matrix = np.zeros((len(problems), len(kcs)), dtype=int)

for i, problem in enumerate(problems):
    new_problem_id = problem_id_map[problem]
    problem_df = df_filtered[df_filtered['problem_id'] == problem]
    kcs_of_problem = problem_df['skill_id'].unique()
    q_matrix[new_problem_id, np.isin(kcs, kcs_of_problem)] = 1

# Save the updated Q-matrix as a CSV file
np.savetxt('q_matrix.csv', q_matrix, fmt='%d', delimiter=',')

# Save the filtered training and testing dataset with new IDs as CSV files
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
