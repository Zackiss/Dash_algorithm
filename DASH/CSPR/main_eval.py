from itertools import cycle
from longling.ML.toolkit.monitor import EMAValue

from DASH.CSPR.KSS.agent import KSSAgent
from DASH.CSPR.KSS.env import KSSEnv


def main_eval(agent: KSSAgent, env: KSSEnv,
              max_steps: int = None, max_episode_num: int = None, values: dict = None):
    """
    the major process is:
        env.step(an action)
        an action is produced by an agent. Currently, an action is just
        trying to randomly extract a knowledge point from the ListSpace
        the env.step(an action) will return observation and reward

    Args:
        agent:
        env: gym environment
        max_steps:
            When max_steps is set (i.e., max_steps is not None):
            at each episode, the agent will interactive at maximum of max_steps with environments.
            When max_steps is not set (i.e., max_steps is None): the episode will last until
            the environment return done=True
        max_episode_num:
            max_episode_num should be set when environment is the type of infinity
        values

    Returns:

    """
    episode = 0

    if values is None:
        values = {"Episode": EMAValue(["Reward"])}

    loop = cycle([1]) if max_episode_num is None else range(max_episode_num)

    for t in loop:
        if max_episode_num is not None and episode >= max_episode_num:
            break

        # It is worth to notice that, the agent produce "action", here the action should be either learning
        # path (n_step) or learning item (step), and give it to env, after judgement, env will produce the result of
        # next learning item recommendation, be either learning path or learning item. Then the agent will receive the
        # decided new "actions" and save it to a certain place in the agent class for preparing the next "action"
        # fetch by the env
        try:
            learner_profile = env.begin_episode()
            agent.begin_episode(learner_profile)
            episode += 1

            print("episode [%s]: %s" % (episode, env.render()))

        except StopIteration:  # pragma: no cover
            break

        # recommend and learn
        # generate a learning path step by step
        assert max_steps is not None, "Please set up max step of recommendation"
        for _step in range(max_steps):
            try:
                learning_cell = agent.step()
                learning_cell = learning_cell.detach().numpy()

            except StopIteration:  # pragma: no cover
                break
            # noinspection PyTupleAssignmentBalance
            state, reward, done, _ = env.step(learning_cell)

            print("step [%s]: agent -|%s|-> env, env state %s" % (_step, learning_cell, env.render()))
            print("step [%s]: observation: %s, reward: %s" % (_step, state, reward))

            agent.observe(state, reward, done, t)
            if done:
                break

        # test the learner to see the learning effectiveness
        state, reward, done, info = env.end_episode()
        agent.end_episode(state, reward, done, info)

        print("episode [%s] - learning path: %s" % (episode, agent.extract_learning_path()))
        print("episode [%s] - total reward: %s" % (episode, reward))
        print("episode [%s]: %s" % (episode, env.render()))

        values["Episode"].update("Reward", reward)

        env.reset()
