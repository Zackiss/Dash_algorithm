import os

from DASH.LPKT import LPKT
from DASH.LPKT.data_loader import StudentSequenceDataset

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split

from DASH.encoder import Encoder


def KT_train(KT_filepath, encoder: Encoder, batch_size, dropout=0.2):
    # set up Knowledge Tracing Model
    KT = LPKT(
        n_at=encoder.num_time,
        n_it=encoder.num_time,
        n_exercise=encoder.num_topic,
        n_question=encoder.num_problem,
        d_a=encoder.outcome_dim,
        d_e=encoder.problem_dim,
        d_k=encoder.time_dim,
        q_matrix=encoder.q_matrix,
        batch_size=batch_size,
        dropout=dropout
    )

    # train on LPKT model for knowledge tracing if necessary
    if not os.path.exists(KT_filepath):
        # DONE prepare data for LPKT training
        dataset = StudentSequenceDataset('student_log_kt_1000.csv', encoder=encoder, max_length=30)
        train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        batch_size = 32  # Adjust according to your needs
        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        KT.train(train_dataloader, test_dataloader, epoch=2)
        KT.save(KT_filepath)

    return KT
