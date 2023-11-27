import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LPKTNet(nn.Module):
    def __init__(self, n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, dropout=0.2):
        super(LPKTNet, self).__init__()
        self.d_k = d_k
        self.d_a = d_a
        self.d_e = d_e
        self.q_matrix = q_matrix
        self.n_question = n_question

        self.at_embed = nn.Embedding(n_at + 10, d_k)
        torch.nn.init.xavier_uniform_(self.at_embed.weight)
        self.it_embed = nn.Embedding(n_it + 10, d_k)
        torch.nn.init.xavier_uniform_(self.it_embed.weight)
        self.e_embed = nn.Embedding(n_exercise + 10, d_k)
        torch.nn.init.xavier_uniform_(self.e_embed.weight)

        # the MLP embedding of the learning cell
        self.linear_1 = nn.Linear(d_a + d_e + d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        self.linear_2 = nn.Linear(4 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        self.linear_3 = nn.Linear(4 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_3.weight)
        self.linear_4 = nn.Linear(3 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_4.weight)
        self.linear_5 = nn.Linear(d_e + d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_5.weight)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, e_data, at_data, a_data, it_data):
        """
        Input the data of learning sequence.

        Args:
            e_data:
            at_data:
            a_data:
            it_data:

        Returns:

        """
        batch_size, seq_len = e_data.size(0), e_data.size(1)
        e_embed_data = self.e_embed(e_data)
        at_embed_data = self.at_embed(at_data)
        it_embed_data = self.it_embed(it_data)
        a_data = a_data.view(-1, 1).repeat(1, self.d_a).view(batch_size, -1, self.d_a)
        h_pre = nn.init.xavier_uniform_(torch.zeros(self.n_question + 1, self.d_k)).repeat(batch_size, 1, 1).to(device)
        h_tilde_pre = None
        all_learning = self.linear_1(torch.cat((e_embed_data, at_embed_data, a_data), 2))
        learning_pre = torch.zeros(batch_size, self.d_k).to(device)

        pred = torch.zeros(batch_size, seq_len).to(device)

        for t in range(0, seq_len - 1):
            e = e_data[:, t]
            # q_e: (bs, 1, n_skill)
            q_e = self.q_matrix[e].view(batch_size, 1, -1)
            it = it_embed_data[:, t]

            # Learning Module
            if h_tilde_pre is None:
                h_tilde_pre = q_e.bmm(h_pre).view(batch_size, self.d_k)
            learning = all_learning[:, t]
            learning_gain = self.linear_2(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
            learning_gain = self.tanh(learning_gain)
            gamma_l = self.linear_3(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
            gamma_l = self.sig(gamma_l)
            LG = gamma_l * ((learning_gain + 1) / 2)
            LG_tilde = self.dropout(q_e.transpose(1, 2).bmm(LG.view(batch_size, 1, -1)))

            # Forgetting Module
            # h_pre: (bs, n_skill, d_k)
            # LG: (bs, d_k)
            # it: (bs, d_k)
            n_skill = LG_tilde.size(1)
            gamma_f = self.sig(self.linear_4(torch.cat((
                h_pre,
                LG.repeat(1, n_skill).view(batch_size, -1, self.d_k),
                it.repeat(1, n_skill).view(batch_size, -1, self.d_k)
            ), 2)))
            h = LG_tilde + gamma_f * h_pre

            # Predicting Module
            h_tilde = self.q_matrix[e_data[:, t + 1]].view(batch_size, 1, -1).bmm(h).view(batch_size, self.d_k)
            y = self.sig(self.linear_5(torch.cat((e_embed_data[:, t + 1], h_tilde), 1))).sum(1) / self.d_k
            pred[:, t + 1] = y

            # prepare for next prediction
            learning_pre = learning
            h_pre = h
            h_tilde_pre = h_tilde

        return pred

    def obtain_h_from_h_batch(self, e_batch, at_batch, a_batch, it_batch):
        """
        Update the hidden states h_pre_batch based on the batch inputs.

        Args:
            e_batch: The batch of input exercises. Size: (batch_size, context_len)
            at_batch: The batch of answer times. Size: (batch_size, context_len)
            a_batch: The batch of input answers. Size: (batch_size, context_len, dim_a)
            it_batch: The batch of interval times. Size: (batch_size, context_len)

        Returns:
            All hidden states h_all. Size: (batch_size, context_len, h_dim)
        """
        batch_size, context_len, _ = e_batch.size()
        reward = None
        # h_pre_batch: The previous hidden states batch. Size: (batch_size, context_len, h_dim)
        h_pre_batch = torch.zeros(batch_size, context_len, self.h_dim).to(device)

        for i in range(context_len):
            e = e_batch[:, i]  # Exercise tensor for current time step. Size: (batch_size,)
            at = at_batch[:, i]  # Answer time tensor for current time step. Size: (batch_size,)
            a = a_batch[:, i]  # Answer tensor for current time step. Size: (batch_size, dim_a)
            it = it_batch[:, i]  # Interval time tensor for current time step. Size: (batch_size,)

            q_e = self.q_matrix[e].unsqueeze(1)  # Exercise embedding tensor. Size: (batch_size, 1, q_dim)
            it_embed = self.it_embed(it)  # Interval time embedding tensor. Size: (batch_size, it_dim)

            all_learning = self.linear_1(
                torch.cat((self.e_embed(e), self.at_embed(at), a), dim=2)
            )  # Concatenated input tensor. Size: (batch_size, 1, all_dim)

            learning_pre = torch.zeros(batch_size, self.d_k).to(device)  # Learning pre tensor. Size: (batch_size, d_k)
            learning_gain = self.linear_2(torch.cat((learning_pre, it_embed, all_learning), dim=1))
            learning_gain = self.tanh(learning_gain)  # Learning gain tensor. Size: (batch_size, d_k)
            gamma_l = self.linear_3(torch.cat((learning_pre, it_embed, all_learning), dim=1))
            gamma_l = self.sig(gamma_l)  # Gamma_l tensor. Size: (batch_size, d_k)
            LG = gamma_l * ((learning_gain + 1) / 2)  # LG tensor. Size: (batch_size, d_k)
            LG_tilde = self.dropout(q_e.transpose(1, 2).bmm(LG.unsqueeze(2))).squeeze(2)
            # LG_tilde tensor. Size: (batch_size, q_dim)

            n_skill = LG_tilde.size(1)
            gamma_f = self.sig(
                self.linear_4(
                    torch.cat((h_pre_batch, LG.repeat(1, n_skill).view(batch_size, -1, self.d_k),
                               it_embed.repeat(1, n_skill).view(batch_size, -1, self.d_k)), dim=2)
                )
            )  # Gamma_f tensor. Size: (batch_size, n_skill, d_k)
            h = LG_tilde + gamma_f * h_pre_batch[:, i]  # Updated h tensor at current time step
            h_pre_batch[:, i] = h

            last_h = h_pre_batch[:, -1]  # Last h in the batch across time steps
            next_h = torch.cat((h_pre_batch[:, 1:], torch.zeros(batch_size, 1, self.h_dim).to(device)), dim=1)
            reward = next_h - last_h.unsqueeze(1)
            reward[:, -1] = torch.zeros(batch_size, self.h_dim).to(device)

        gamma = 0.9  # Discount factor (you can adjust this value)
        reward_np = reward.cpu().numpy()  # Convert reward tensor to numpy array
        return_to_go = np.zeros_like(reward_np)

        for b in range(batch_size):
            return_to_go[b] = discount_cum_sum(reward_np[b], gamma)
        return_to_go_tensor = torch.from_numpy(return_to_go).to(device)

        return h_pre_batch, return_to_go_tensor


def discount_cum_sum(x, gamma):
    disc_cum_sum = np.zeros_like(x)
    disc_cum_sum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cum_sum[t] = x[t] + gamma * disc_cum_sum[t + 1]
    return disc_cum_sum
