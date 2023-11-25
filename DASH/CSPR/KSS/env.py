from copy import deepcopy

import gym
import networkx as nx
import random

import numpy as np
from pprint import pformat

from gym.spaces.space import T_cov

from DASH.CSPR.KSS.meta.learner import LearnerGroup, Learner
from DASH.CSPR.KES.reward import episode_reward
from DASH.CSPR.KSS.utils import load_environment_parameters
from DASH.LPKT import LPKT


# TODO cut the cell to item vector
def convert_cell_to_item(learning_cell):
    return None


class KSSEnv(gym.Env):
    def __repr__(self):
        return pformat(self.parameters)

    def __init__(self, knowledge_tracing: LPKT, start_state, seed=None, initial_step=20):
        # load parameters from meta data file
        parameters = load_environment_parameters()

        # knowledge structure parallel to the file meta_data/knowledge_structure.csv
        # representation of connections in knowledge graph (KG)
        self.knowledge_structure = parameters["knowledge_structure"]
        self.knowledge_tracing_model = knowledge_tracing

        self.learners = LearnerGroup(self.knowledge_structure,
                                     self.knowledge_tracing_model,
                                     start_state,
                                     seed=seed
                                     )
        self._order_ratio = parameters["configuration"]["order_ratio"]
        self._review_times = parameters["configuration"]["review_times"]
        self._learning_order = parameters["learning_order"]

        self._topo_order = list(nx.topological_sort(self.knowledge_structure))
        self._initial_step = parameters["configuration"]["initial_step"] if initial_step is None else initial_step

        self._learner = None
        self._initial_score = None
        self._exam_reduce = "sum" if parameters["configuration"].get("exam_sum", True) else "ave"

    @property
    def parameters(self) -> dict:
        return {
            "knowledge_structure": self.knowledge_structure,
            "action_space": self.action_space,
        }

    def learn_and_test(self, learner: Learner, learning_cell):
        # TODO cut the cell to item
        learning_item = convert_cell_to_item(learning_cell)
        learner.learn(learning_item)
        test_item = learning_item
        score = learner.response(test_item)
        return test_item, score

    def _exam(self, learner: Learner, detailed=False, reduce=None) -> (dict, int, float):
        if reduce is None:
            reduce = self._exam_reduce
        knowledge_response = {}
        for test_knowledge in learner.target:
            # TODO convert knowledge to item correctly
            item = self.test_item_base.knowledge2item[test_knowledge]
            # Using learner.response() to predict the score given by knowledge tracing model
            knowledge_response[test_knowledge] = [item.id, learner.response_item(item)]
        if detailed:
            return knowledge_response
        elif reduce == "sum":
            return np.sum([v for _, v in knowledge_response.values()])
        elif reduce in {"mean", "ave"}:
            return np.average([v for _, v in knowledge_response.values()])
        else:
            raise TypeError("unknown reduce type %s" % reduce)

    def begin_episode(self, *args, **kwargs):
        """
        At beginning of each episode, the learner will be extracted from learners.
        The next learner will be extracted, and we will record the init score, which
        will be used on calculating the episode reward, which is the effectiveness
        of a learning session
        """
        self._learner = next(self.learners)
        self._initial_score = self._exam(self._learner)
        return self._learner.profile, self._exam(self._learner, detailed=True)

    def end_episode(self, *args, **kwargs):
        """
        At the end of episode, the exam will be taken on the current learner in episode
        The score will be used on calculating the episode reward, which is the effectiveness
        of a learning session
        """
        observation = self._exam(self._learner, detailed=True)
        initial_score, self._initial_score = self._initial_score, None
        final_score = self._exam(self._learner)
        reward = episode_reward(initial_score, final_score, len(self._learner.target))
        done = final_score == len(self._learner.target)
        info = {"initial_score": initial_score, "final_score": final_score}
        self._learner = None

        return observation, reward, done, info

    def step(self, learning_cell, *args, **kwargs):
        """
        Normally, an environment step describe an action (leaning_item_id) was taken
        by the learner, and will return the new environment state after the action
        and the reward of this action. They will be used for the optimization of RL
        The reward here is defined by final exam mark - begin exam mark

        Args:
            learning_cell:
            *args:
            **kwargs:

        Returns:

        """
        a = self._exam(self._learner)
        state = self.learn_and_test(self._learner, learning_cell)
        b = self._exam(self._learner)

        return state, b - a, b == len(self._learner.target), None

    def reset(self, seed=None, options=None):
        self._learner = None

    def render(self):
        if self._learner is not None:
            return "target: %s, state: %s" % (
                self._learner.target, self._exam(self._learner)
            )
        else:
            return "Learner not specify"

