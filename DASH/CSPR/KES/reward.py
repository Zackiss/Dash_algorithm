def episode_reward(initial_score, final_score, full_score) -> (int, float):
    """
    the sum of all the rewards for each timestep in an episode
    the episode reward here represent the effectiveness of a learning session
    multiple learning sessions compose a learning procedure

    Args:
        initial_score:
        final_score:
        full_score:

    Returns:

    """
    delta = final_score - initial_score
    normalize_factor = full_score - initial_score

    return delta / normalize_factor
