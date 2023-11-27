import json
import os

import pandas as pd
import numpy as np


class Encoder:
    def __init__(self):
        self.problem_dim = 128
        self.exercise_mean = 0
        self.exercise_div = 1
        self.problem_enc = self.encode_problems()
        self.q_matrix = self.gen_q_matrix()
        self.num_problem = self.q_matrix.shape[0]
        self.num_topic = self.q_matrix.shape[1]
        self.outcome_dim = 16
        self.time_dim = 128
        self.num_time = self.time_dim
        self.max_time = 500
        self.cell_size = self.num_time * 2 + 1 + self.num_problem

    def gen_q_matrix(self):
        if os.path.exists('q_matrix.npy'):
            q_matrix = np.load('q_matrix.npy')
        else:
            # Read the CSV file
            df = pd.read_csv('junyi_Exercise_table.csv')
            with open("graph_vertex.json") as f:
                ku_dict = json.load(f)

            # Extract unique problem names and topics
            problem_names = df['problem_name'].unique()
            topics = df['topic'].unique()
            m_topic = len(topics)

            # Create a dictionary to map topics to their corresponding indices
            topic_to_idx = {topic: idx for idx, topic in enumerate(topics)}
            problem_name_to_idx = {name: ku_dict[str(name)] for idx, name in enumerate(problem_names)}

            # Create the Q matrix and initialize with a default value
            q_matrix = np.full((self.problem_dim, m_topic), 0.3)

            # Update the Q matrix based on the relationship between problem names and topics
            for _, row in df.iterrows():
                problem_name = row['problem_name']
                topic = row['topic']
                q_matrix[problem_name_to_idx[problem_name]][topic_to_idx[topic]] = 1

            np.save('q_matrix.npy', q_matrix)

        return q_matrix

    def encode_problems(self):
        if os.path.exists("problem_id_embedding.npy"):
            problem_id_embedding = np.load('problem_id_embedding.npy')
        else:
            # Load the CSV dataset
            df = pd.read_csv('junyi_Exercise_table.csv')
            # Extract unique problem IDs and sort them in increasing order
            problem_ids = sorted(df['problem_id'].unique())
            num_problems = len(problem_ids)
            problem_id_embedding = np.random.normal(
                self.exercise_mean, self.exercise_div, size=(num_problems, self.problem_dim)
            )
            # Save the embeddings as a NumPy array
            np.save('problem_id_embedding.npy', problem_id_embedding)
        return problem_id_embedding

    def encode_sequence(self, problem_id, outcome, time_spent, interval_time, split=True):
        problem_id_vector = self.problem_enc[problem_id]

        # Encode the outcome as a vector
        outcome_vector = np.zeros(self.outcome_dim)
        if outcome == 1:
            outcome_vector = np.ones(self.outcome_dim)

        # Generate time_bins using logarithmic spacing
        time_bins = np.logspace(0, np.log2(self.max_time), self.time_dim, base=2.0)

        at_encoded = np.digitize(time_spent, time_bins, right=True) - 1  # Subtract 1 to match indexing
        at_vector = np.eye(self.time_dim)[at_encoded]
        # at_vector shape: (time_dim,)

        it_encoded = np.digitize(interval_time, time_bins, right=True) - 1  # Subtract 1 to match indexing
        it_vector = np.eye(self.time_dim)[it_encoded]
        # it_vector shape: (time_dim,)

        if split:
            # many row vectors
            return problem_id_vector, outcome_vector, at_vector, it_vector
        else:
            # concatenate all the vectors, a row vector
            combined_vector = np.concatenate(
                [problem_id_vector, outcome_vector, at_vector, it_vector]
            )

            return combined_vector

    def encode_deck(self, problem_ids, outcomes, times_spent, interval_times):
        problem_id_vectors = np.array([self.problem_enc[problem_id] for problem_id in problem_ids])
        # problem_id_vectors shape: (num_problems, problem_dim)

        outcome_vectors = np.array(
            [np.zeros(self.outcome_dim) if outcome == 0 else np.ones(self.outcome_dim) for outcome in outcomes])
        # outcome_vectors shape: (num_outcomes, outcome_dim)

        time_bins = np.logspace(0, np.log2(self.max_time), self.time_dim, base=2.0)

        at_encoded = np.digitize(times_spent, time_bins)
        at_vectors = np.eye(self.time_dim + 1)[at_encoded]
        at_vectors = at_vectors[:, :-1]
        # at_vectors shape: (num_times_spent, time_dim)

        it_encoded = np.digitize(interval_times, time_bins)
        it_vectors = np.eye(self.time_dim + 1)[it_encoded]
        it_vectors = it_vectors[:, :-1]
        # it_vectors shape: (num_interval_times, time_dim)

        return problem_id_vectors, outcome_vectors, at_vectors, it_vectors

