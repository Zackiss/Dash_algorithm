import os
import numpy as np
import torch
from transformers import TrainingArguments, Trainer, DecisionTransformerConfig

from DASH.CSPR.KSS.agent import KSSAgent
from DASH.CSPR.KSS.env import KSSEnv
from DASH.CSPR.main_eval import main_eval
from DASH.DT.decision_transformer import DecisionTransformer, DTDataCollator
from DASH.LPKT import LPKT
from DASH.dataset.test_data_gen import test_data_gen, test_conf

conf = test_conf()

# TODO prepare data for LPKT training
data = test_data_gen

KT_filepath = "./KT_params"
DT_filepath = "./DT_params"

# set up Knowledge Tracing Model
KT = LPKT(
    n_at=conf[0],
    n_it=conf[1],
    n_exercise=conf[2],
    n_question=conf[3],
    d_a=16,
    d_e=32,
    d_k=32,
    q_matrix=conf[4],
    batch_size=conf[5],
    dropout=0.2
)

# set up Decision Transformer
collator = DTDataCollator(dataset["train"])
config = DecisionTransformerConfig(
    state_dim=collator.state_dim,
    act_dim=collator.act_dim
)
DT = DecisionTransformer(config)


# train Decision Transformer if necessary
if not os.path.exists(KT_filepath):
    training_args = TrainingArguments(
        output_dir=DT_filepath,
        remove_unused_columns=False,
        num_train_epochs=120,
        per_device_train_batch_size=64,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        optim="adamw_torch",
        max_grad_norm=0.25,
    )

    trainer = Trainer(
        model=DT,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=collator,
    )

    trainer.train()
    DT.save_pretrained(DT_filepath)
# load the Decision Transformer model
DT.from_pretrained(DT_filepath)


# train on LPKT model for knowledge tracing if necessary
if not os.path.exists(KT_filepath):
    KT.train(data, test_data=data, epoch=2)
    KT.save(KT_filepath)
# load the LPKT model
KT.load(KT_filepath)


# define the desired mean and variance for the random start state
start_state_mean, start_state_variance = 0.0, 0.1
# generate the random start state
start_state = torch.randn(DT.config.state_dim) * start_state_variance + start_state_mean

# set up Knowledge-graph based Recommendation System Environment
# TODO prepare the dataset of KSS meta-data
env = KSSEnv(
    knowledge_tracing=KT,
    start_state=start_state
)

# eval on KSS framework for reinforcement learning
main_eval(
    KSSAgent(
        decision_model=DT,
        start_state=start_state,
        # TODO when collecting data from main_eval for transformer training, the following params are prohibited,
        #  so as to all "dataset" code mentioned above
        state_mean=collator.state_mean.astype(np.float32),
        state_std=collator.state_std.astype(np.float32)
    ),
    env,
    20, 4000
)
