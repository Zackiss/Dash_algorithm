import torch


class KSSAgent(object):
    def __init__(self, decision_model, start_state, state_mean, state_std, reward_scale=1000.0, target_return=3.6):
        self.decision_model = decision_model
        self.reward_scale = reward_scale
        self.state_mean = state_mean
        self.state_std = state_std
        self.target_return = torch.tensor(target_return).float().reshape(1, 1)
        self.states = torch.from_numpy(start_state).reshape(1, self.decision_model.config.state_dim).float()
        self.rewards = torch.zeros(0).float()
        self.time_steps = torch.tensor(0).reshape(1, 1).long()
        self.actions = torch.zeros((0, self.decision_model.config.act_dim)).float()

    def begin_episode(self, *args, **kwargs):
        pass

    def end_episode(self, observation, reward, done, info):
        pass

    def observe(self, state, reward, done, t):
        """
        Agent observe is used for preparing the get_action (agent.step)
        It record the observation, reward to the agent so that the step will
        generate the correct step

        Args:
            state:
            reward:
            done:
            info:

        Returns:

        """
        cur_state = torch.from_numpy(state).reshape(1, self.decision_model.config.state_dim)
        self.states = torch.cat([self.states, cur_state], dim=0)
        self.rewards[-1] = reward

        pred_return = self.target_return[0, -1] - (reward / self.reward_scale)
        self.target_return = torch.cat([self.target_return, pred_return.reshape(1, 1)], dim=1)
        self.time_steps = torch.cat([self.time_steps, torch.ones((1, 1)).long() * (t + 1)], dim=1)

        # TODO In the decision transformer training process, you need to record the necessary rewards and so on
        #  to build the dataset

    def step(self):
        # This implementation does not condition on past rewards
        self.actions = torch.cat([self.actions, torch.zeros((1, self.decision_model.config.act_dim))], dim=0)
        self.rewards = torch.cat([self.rewards, torch.zeros(1)])

        self.states = (self.states - self.state_mean) / self.state_std

        states = self.states.reshape(1, -1, self.decision_model.config.state_dim)
        actions = self.actions.reshape(1, -1, self.decision_model.config.act_dim)
        returns_to_go = self.target_return.reshape(1, -1, 1)
        time_steps = self.time_steps.reshape(1, -1)

        states = states[:, -self.decision_model.config.max_length:]
        actions = actions[:, -self.decision_model.config.max_length:]
        returns_to_go = returns_to_go[:, -self.decision_model.config.max_length:]
        time_steps = time_steps[:, -self.decision_model.config.max_length:]
        padding = self.decision_model.config.max_length - states.shape[1]
        # pad all tokens to sequence length
        attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
        attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
        states = torch.cat([torch.zeros((1, padding, self.decision_model.config.state_dim)), states], dim=1).float()
        actions = torch.cat([torch.zeros((1, padding, self.decision_model.config.act_dim)), actions], dim=1).float()
        returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
        time_steps = torch.cat([torch.zeros((1, padding), dtype=torch.long), time_steps], dim=1)

        # state_preds, action_preds, return_preds = self.decision_model.original_forward(
        #     states=states,
        #     actions=actions,
        #     rewards=self.rewards,
        #     returns_to_go=returns_to_go,
        #     timesteps=time_steps,
        #     attention_mask=attention_mask,
        #     return_dict=False,
        # )

        # TODO in the decision model training process, you should recommend the next action directly from the dataset

        self.actions[-1] = action_preds[0, -1]

        return action_preds[0, -1]

    # TODO implement method for converting the output of agent.step to item
    def action_to_item(self):
        pass

    def extract_learning_path(self):
        return self.actions

    def tune(self, *args, **kwargs):
        pass
