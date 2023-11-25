import uuid

import numpy as np
import math

import networkx as nx
from DASH.CSPR.KES.KS import influence_control
from DASH.LPKT import LPKT


class LearningModel(object):
    def __init__(self, state, learning_target, knowledge_structure, knowledge_tracing: LPKT, last_visit=None):
        self._state = state
        self._target = learning_target
        self._ks = knowledge_structure
        self._ks_last_visit = last_visit
        self.knowledge_tracing = knowledge_tracing

    def step(self, state, learning_cell):
        # learning item is a vector
        # TODO convert learning_cell vector to its knowledge
        # TODO use KT and learning cell to predict the state
        if self._ks_last_visit is not None:
            if knowledge not in influence_control(
                    self._ks, state, self._ks_last_visit, allow_shortcut=False, target=self._target,
            )[0]:
                return state
        self._ks_last_visit = knowledge

        # capacity growth function
        discount = math.exp(sum([(5 - state[node]) for node in self._ks.predecessors(knowledge)] + [0]))
        ratio = 1 / discount
        inc = (5 - state[knowledge]) * ratio * 0.5

        def _promote(_ind, _inc):
            state[_ind] += _inc
            if state[_ind] > 5:
                state[_ind] = 5
            for node in self._ks.successors(_ind):
                _promote(node, _inc * 0.5)

        _promote(knowledge, inc)
        return state


class Learner(object):
    def __init__(self,
                 initial_state,
                 knowledge_structure: nx.DiGraph,
                 knowledge_tracing: LPKT,
                 learning_target: set,
                 _id=None,
                 seed=None):
        self.id = self.set_id(_id)

        self.learning_model = LearningModel(
            initial_state,
            learning_target,
            knowledge_structure,
            knowledge_tracing
        )

        self.structure = knowledge_structure
        self._state = initial_state
        self._target = learning_target
        self._logs = []
        self.random_state = np.random.RandomState(seed)

    @classmethod
    def set_id(cls, _id=None):
        return _id if _id is not None else uuid.uuid1()

    def update_logs(self, logs):
        self._logs = logs

    @property
    def profile(self):
        return {
            "id": self.id,
            "target": self.target
        }

    def learn(self, learning_cell):
        """learn a new learning cell, which can result in state changing"""
        # learning cell is an item embedding vector
        self._state = self.learning_model.step(self._state, learning_cell)

    @property
    def state(self):
        return self._state

    def response(self, test_cell) -> ...:
        """
        Give the response to the test_cell

        Args:
            test_cell: a test cell
        """
        # learning item is an item embedding vector
        # TODO convert test_cell vector to its knowledge
        return self._state[test_cell.knowledge]

    def response_item(self, learning_item):
        # TODO use learning_item and h (learner._state) to predict the possibility of
        #  correctness of the given learning item
        return probability of correctness

    @property
    def target(self):
        return self._target


class LearnerGroup(object):
    """
    Examples:
        >> learners = [MetaLearner(i) for i in range(10)]
        >> mflg = MetaFiniteLearnerGroup(learners)
        >> mflg.__len__()
        10
        >> isinstance(mflg.__getitem__[0],MetaLearner)
        True
        >> isinstance(mflg.sample(),MetaLearner)
        True
    """

    def __init__(self, knowledge_structure, knowledge_tracing: LPKT, initial_state, seed=None):
        super(LearnerGroup, self).__init__()
        self.knowledge_structure = knowledge_structure
        self.knowledge_tracing = knowledge_tracing
        self.random_state = np.random.RandomState(seed)
        self.initial_state = initial_state

    def __next__(self):
        knowledge = self.knowledge_structure.nodes
        return Learner(
            # the initial state of learner
            self.initial_state,
            # the related knowledge structure
            self.knowledge_structure,
            # knowledge tracing model
            self.knowledge_tracing,
            # the set of learning target
            set(self.random_state.choice(len(knowledge), self.random_state.randint(3, len(knowledge)))),
        )
