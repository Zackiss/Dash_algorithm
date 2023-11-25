import logging
import cupy as np
import pickle
from tqdm import tqdm
from collections import namedtuple
from collections import defaultdict

hyper_para = namedtuple("hyperparameters", ["r", "D", "deltaT", "S", "lambda_U_1", "lambda_U", "lambda_P", "lambda_S"])
default_hyper = hyper_para(6, 2, 1, 5, 0.01, 2, 2, 0.01)


def init_parameters(stu_num, prob_num, know_num, time_window_num):
    """
    Store the hidden vectors for students and problems.
    Note that the problems' latent vector is in R^know_num.

    Returns:
        u_latent:
            The students' hidden vectors, trained to represent the students' mastery degree of
            all concepts. The size should be in [ student_num * concept_num ]

    """
    u_latent = np.random.normal(0.5, 0.01, size=(time_window_num, stu_num, know_num))
    i_latent = 0.1 * np.random.uniform(0, 1, size=(prob_num, know_num))  # problems' latent vector(V)
    alpha = np.random.uniform(0, 1, size=stu_num)
    B = 0.01 * np.random.normal(0, 1, size=prob_num)
    return u_latent, i_latent, alpha, B


def stu_curve(u_latent, alpha, r, D, deltaT, S, time_freq):
    """
    Calculate the learning curve and memory loss curve for students in u_latent.
    The calculation performs based on students' hidden vectors.

    Args:
        u_latent:
            The students' hidden vectors, trained to represent the students' mastery degree of
            all concepts. The size should be in [ student_num * concept_num ]
        alpha:
            a * learning_curve + (1 - a) * forget_curve
            Used to balance the two, will be trained and updated.
            This will be the key concept on personalization recommendation.
        r, D:
            Control the growth rate of learning_curve
        deltaT, S:
            Some hyperparameters for forget_curve
        time_freq:
            The frequency of students' answering. Each entry records the num of questions students
            worked on for a certain concept (KCs)
            The size should be in [ student_num * concept_num ]
    Returns:

    """
    freq_norm = D * time_freq / (time_freq + r)
    learn_factor = u_latent * freq_norm
    forget_factor = u_latent * np.exp(-deltaT / S)

    # learning and forgetting curve
    pred_u = learn_factor * np.expand_dims(alpha, axis=1) + forget_factor * np.expand_dims(1 - alpha, axis=1)
    return pred_u, freq_norm


class EKPT:
    def __init__(self, q_m, stu_num, prob_num, know_num, time_window_num, args=default_hyper):
        """
        EKPT model with training and testing methods

        Args:
            q_m:
                Q matrix, used for recording the co-relationship between problems and concepts
                it should be in shape [ prob_num * know_num ]
            stu_num, prob_num, know_num:
                Num of students, problems, knowledge components
            time_window_num:
                Num of time windows split from dataset in sequence of time,
                and the time windows' num has been set to 7, be sure to check if it is valid.
                There is a possibility that wrongly set the windows' num make the eval acts like disaster
            args:
                Some other hyperparameters

        """
        super(EKPT, self).__init__()
        self.args = args
        self.q_m = q_m
        self.stu_num, self.prob_num, self.know_num = stu_num, prob_num, know_num
        self.time_window_num = time_window_num
        self.u_latent, self.i_latent, self.alpha, self.B = init_parameters(stu_num, prob_num, know_num, time_window_num)
        # partial order of knowledge in each problem
        self.par_mat = np.zeros(shape=(prob_num, know_num, know_num))
        for i in range(prob_num):
            for o1 in range(know_num):
                if self.q_m[i][o1] == 0:
                    continue
                for o2 in range(know_num):
                    if self.q_m[i][o2] == 0:
                        self.par_mat[i][o1][o2] = 1

        # exercise relation
        self.exer_neigh = (np.dot(self.q_m, self.q_m.transpose()) > 0).astype(int)
        self.time_freq = None

    def train(self, train_data, epoch, lr=0.001, lr_b=0.0001, epsilon=1e-3, init_method='mean') -> ...:
        """
        Optimize the model with gradient descent method.

        Args:
            train_data:
                List of response dataset, length = time_window_num.
                Each element is a list of dictionaries representing user-item interactions.
                Example: [[{'user_id':, 'item_id':, 'score':},...],...]
            epoch:
                Number of training epochs.
            lr, lr_b:
                Learning rate for latent factors and bias term.
            epsilon:
                Convergence threshold.
            init_method:
                Method for initializing student latent factors.

        """
        # train_data(list): response dataset, length = time_window_num,
        # e.g.[[{'user_id':, 'item_id':, 'score':},...],...]
        assert self.time_window_num == len(train_data), 'number of time windows conflicts'
        u_latent, i_latent = np.copy(self.u_latent), np.copy(self.i_latent)
        alpha, B = np.copy(self.alpha), np.copy(self.B)
        # mean score of each student in train_data
        sum_score = np.zeros(shape=self.stu_num)
        sum_count = np.zeros(shape=self.stu_num)

        # knowledge frequency in each time window
        self.time_freq = np.zeros(shape=(self.time_window_num, self.stu_num, self.know_num))
        for t in range(self.time_window_num):
            for record in train_data[t]:
                user, item, rating = record['user_id'], record['item_id'], record['score']
                self.time_freq[t][user][np.where(self.q_m[item] == 1)[0]] += 1
                sum_score[user] += rating
                sum_count[user] += 1

        # initialize student latent with mean score
        if init_method == 'mean':
            u_latent = np.random.normal(20 * np.expand_dims(sum_score / (sum_count + 1e-9), axis=1) / self.know_num,
                                        0.01, size=(self.time_window_num, self.stu_num, self.know_num))

        for iteration in range(epoch):
            u_latent_tmp, i_latent_tmp = np.copy(u_latent), np.copy(i_latent)
            alpha_tmp, B_tmp = np.copy(alpha), np.copy(B)
            i_gradient = np.zeros(shape=(self.prob_num, self.know_num))
            b_gradient = np.zeros(shape=self.prob_num)
            alpha_gradient = np.zeros(shape=self.stu_num)
            for t in range(self.time_window_num):
                u_gradient_t = np.zeros(shape=(self.stu_num, self.know_num))
                record_num_t = len(train_data[t])
                users = [record['user_id'] for record in train_data[t]]
                items = [record['item_id'] for record in train_data[t]]
                ratings = [record['score'] for record in train_data[t]]

                pred_R = [np.dot(u_latent[t][users[i]], i_latent[items[i]]) - B[items[i]] for i in range(record_num_t)]
                pred_u, freq_norm = stu_curve(u_latent, alpha, self.args.r, self.args.D, self.args.deltaT, self.args.S,
                                              self.time_freq)  # both shape are (time_window_num, stu_num, know_num)
                for i in range(record_num_t):
                    user, item, rating = users[i], items[i], ratings[i]
                    R_diff = pred_R[i] - rating
                    b_gradient[item] -= R_diff
                    u_gradient_t[user] += R_diff * i_latent[item]
                    i_gradient[item] += R_diff * u_latent[t][user] + self.args.lambda_S * i_latent[item]
                    i_gradient[item] -= self.args.lambda_S * np.sum(
                        np.expand_dims(self.exer_neigh[item], axis=1) * i_latent, axis=0) / sum(self.exer_neigh[item])
                    if t == 0:
                        u_gradient_t[user] += self.args.lambda_U_1 * u_latent[0][user]
                    else:
                        u_gradient_t[user] += self.args.lambda_U * (u_latent[t][user] - pred_u[t - 1][user])
                        alpha_gradient[user] += np.dot(pred_u[t - 1][user] - u_latent[t][user], u_latent[t][user] * (
                                freq_norm[t - 1][user] - np.exp(-self.args.deltaT / self.args.S)))
                    if t < self.time_window_num - 1:
                        u_gradient_t[user] += self.args.lambda_U * (pred_u[t][user] - u_latent[t + 1][user]) * (
                                alpha[user] * freq_norm[t][user] + (1 - alpha[user]) * np.exp(
                            - self.args.deltaT / self.args.S))
                    o1, o2 = np.where(self.par_mat[item] == 1)
                    for j in range(len(o1)):
                        i_gradient[item][o1[j]] -= self.args.lambda_P * 0.5 * (1 - np.tanh(
                            0.5 * (i_latent[item][o1[j]] - i_latent[item][o2[j]])))
                        i_gradient[item][o2[j]] += self.args.lambda_P * 0.5 * (1 - np.tanh(
                            0.5 * (i_latent[item][o1[j]] - i_latent[item][o2[j]])))
                u_latent[t] -= lr * u_gradient_t
            i_latent -= lr * i_gradient
            B -= lr_b * b_gradient
            alpha = np.clip(alpha - lr * alpha_gradient, 0, 1)
            change = max(np.max(np.abs(u_latent - u_latent_tmp)), np.max(np.abs(i_latent - i_latent_tmp)),
                         np.max(np.abs(alpha - alpha_tmp)), np.max(np.abs(B - B_tmp)))
            print(f"iteration: {iteration}, with change {change}")
            if iteration > 20 and change < epsilon:
                print("training finished!")
                break
        self.u_latent, self.i_latent, self.alpha, self.B = u_latent, i_latent, alpha, B

    def eval(self, test_data) -> tuple:
        """
        Evaluation on the last time windows,
        Determine the distance between the predict score and the real score for all students.

        Args:
            test_data:
                Note that the "item_id" means the "problem_id",
                and the eval is conducted through all recorded elements in test_data

        Returns: The L1 and L2 distance between the real score and the predict score

        """
        test_rmse, test_mae = [], []
        for i in tqdm(test_data, "evaluating"):
            stu, test_id, true_score = i['user_id'], i['item_id'], i['score']
            predict_rating = np.clip(np.dot(self.u_latent[-1][stu], self.i_latent[test_id]) - self.B[test_id], 0, 1)
            test_rmse.append((predict_rating - true_score) ** 2)
            test_mae.append(abs(predict_rating - true_score))
        return np.sqrt(np.average(test_rmse)), np.average(test_mae)

    def prepare_for_transformer_training(self) -> np.ndarray:
        """
        Extract dataset from the model and prepare for transformer training

        Returns:
            time series in this format, a huge matrix: [[reward, status, action], [], ... , []]
            it is worth to notice that every row represents the train dataset of a student's lifetime exp,
            and the "reward" is in length of problem num, "status" and "actions" are in length of KCs num,
            so it should be separately send to the transformer row by row theoretically speaking

        """
        concatenated_data = []
        for sequence_input_index in range(self.time_window_num - 1):
            # add the reward like r(A_{t+1}-A_{t}) from t to T
            reward = np.zeros(shape=(self.stu_num, self.prob_num))
            for t in range(sequence_input_index, self.time_window_num - 1):
                # Extract the latent vectors and biases for all students
                # After that, calculate the rates using matrix multiplication
                student_latents = self.u_latent[t]
                student_next_latents = self.u_latent[t+1]
                prob_biases = self.B.reshape((-1, 1))
                # rates of single student when answering all possible questions, but now for all students in matrix
                rates = np.clip(np.dot(student_latents, self.i_latent.T) - prob_biases, 0, 1)
                next_rates = np.clip(np.dot(student_next_latents, self.i_latent.T) - prob_biases, 0, 1)
                partial_reward = (0.99 ** t) * (next_rates - rates)
                reward += partial_reward

            # status matrix in size of [student_num * KCs_num]
            status = self.u_latent[sequence_input_index]
            action = self.time_freq[sequence_input_index]
            # store to the buffer to be ready for concatenating
            concatenated_data.append(reward)
            concatenated_data.append(status)
            concatenated_data.append(action)
        return np.concatenate(concatenated_data, axis=0)

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump({"U": self.u_latent, "V": self.i_latent, "alpha": self.alpha, "B": self.B}, file)
            logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        with open(filepath, 'rb') as file:
            self.u_latent, self.i_latent, self.alpha, self.B = pickle.load(file).values()
            logging.info("load parameters from %s" % filepath)
