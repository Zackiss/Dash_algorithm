import logging
import cupy as np
from EKPT import EKPT
from KTM.dataset.save_data import load_parsed_data_from_json

# load the Q matrix from dataset
q_m = np.loadtxt("../dataset/q_matrix.csv", dtype=int, delimiter=",")
prob_num, know_num = q_m.shape[0], q_m.shape[1]

# Load parsed training and testing dataset from the JSON file
parsed_train_data = load_parsed_data_from_json('parsed_train_data.json')
parsed_test_data = load_parsed_data_from_json('parsed_test_data.json')

# Get the total number of students
stu_num = max([x['user_id'] for time_window_data in parsed_train_data for x in time_window_data]) + 1
print(f"student Num: {stu_num}")

# Get the time window num, should be 7
time_window_num = len(parsed_train_data)
print(f"time windows Num: {time_window_num}")

logging.getLogger().setLevel(logging.INFO)

cdm = EKPT(q_m, stu_num, prob_num, know_num, time_window_num=time_window_num)

cdm.train(parsed_train_data, epoch=2, lr=0.001, lr_b=0.0001, epsilon=1e-3, init_method='mean')
cdm.save("ekpt.params")

cdm.load("ekpt.params")
rmse, mae = cdm.eval(parsed_test_data)
print("For EKPT, RMSE: %.6f, MAE: %.6f" % (rmse, mae))
