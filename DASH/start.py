import numpy as np
import torch

from DASH.CSPR.KSS.agent import KSSAgent
from DASH.CSPR.KSS.env import KSSEnv
from DASH.CSPR.main_eval import main_eval
from DASH.DT.train import DT_train
from DASH.LPKT.train import KT_train
from DASH.encoder import Encoder

encoder = Encoder()

KT_filepath = "./KT_params"
DT_filepath = "./DT_params"

# train the LPKT model
KT = KT_train(KT_filepath, encoder, batch_size=64)
KT.load(KT_filepath)

# generate the random start state
start_state_mean, start_state_variance = 0.0, 0.1
start_state = torch.randn(encoder.num_topic) * \
              torch.sqrt(torch.tensor(start_state_variance)) + \
              start_state_mean
start_state_zero = torch.zeros(encoder.num_topic)
# KT.lpkt_net.obtain_h_from_h(h_pre, e, at, a, it), h_pre start from pure 0 matrix with state size

DT = DT_train(DT_filepath, KT_model=KT, start_state_zero=start_state_zero,
              state_dim=encoder.num_topic, act_dim=encoder.cell_size)

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
    ),
    env,
    20,
    4000
)
