import csv
import json
import math
from tqdm import tqdm


def _read(source: str, ku_dict: str) -> dict:
    """
    Read the learners' interaction records and classify them by user id and problem id.
    Convert exercise name to exercise id and include discretized interval time.

    Notes
    -----
    Requires significant memory to run this function.
    """

    students = {}

    # convert true and false string in dataset to int
    outcome = {
        "true": 1,
        "false": 0
    }

    with open(ku_dict) as f:
        ku_dict = json.load(f)

    with open(source) as f:
        """
        user_id 0
        exercise 1
        problem_type 2
        problem_number 3
        topic_mode 4
        suggested 5
        review_mode 6
        time_done 7
        time_taken 8
        time_taken_attempts 9
        correct 10
        count_attempts 11
        hint_used 12
        count_hints 13
        hint_time_taken_list 14
        earned_proficiency 15
        points_earned 16
        """
        f.readline()
        for line in tqdm(csv.reader(f), "reading data"):
            student = line[0]  # user_id
            problem = ku_dict[line[1]]  # exercise id
            if outcome[line[12]] == 0:  # hint_used
                correct = outcome[line[10]]  # correct
            else:
                correct = 0
            time_taken = int(line[8])  # time_taken
            time_done = int(line[7])  # time_done

            if student not in students:
                students[student] = []

            students[student].append({
                "exercise_id": problem,
                "exercise_name": line[1],  # exercise name
                "outcome": correct,
                "time_spent": time_taken,
                "time_done": time_done,
                "interval_time": 0  # Placeholder value, will be updated in the next loop iteration
            })

        for student, exercises in students.items():
            # exercises = [{exercise_info}, {exercise_info}]
            exercises.sort(key=lambda x: x["time_done"])
            for i in range(1, len(exercises)):
                interval_time = exercises[i]["time_done"] - exercises[i - 1]["time_done"]
                interval_time_discretized = math.ceil(interval_time / 10 / 60000)
                if interval_time_discretized > 30 * 24 * 60:  # 30 days * 24 hours * 60 minutes
                    interval_time_discretized = 30 * 24 * 60
                exercises[i]["interval_time"] = interval_time_discretized

    return students


def _write(students, target):
    with open(target, 'w', newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(['user_id', 'problem_id', 'exercise_name', 'outcome', 'time_spent', 'time_done', 'interval_time'])

        for student_id, exercises in students.items():
            exercises.sort(key=lambda x: x["time_done"])

            for exercise in exercises:
                writer.writerow([
                    student_id,
                    exercise["exercise_id"],
                    exercise["exercise_name"],
                    exercise["outcome"],
                    exercise["time_spent"],
                    exercise["time_done"],
                    exercise["interval_time"]
                ])


def _frequency(students):
    frequency = {}
    for student_id, problems in tqdm(students.items(), "Calculating frequency"):
        # problems = [{exercise_info}, {exercise_info}]
        frequency[student_id] = len(problems)
    return sorted(frequency.items(), key=lambda x: x[1], reverse=True)


def get_n_most_frequent_students(students, n=None, frequency=None):
    frequency = _frequency(students) if frequency is None else frequency
    __frequency = frequency if n is None else frequency[:n]
    _students = {}
    for student_id, _ in tqdm(__frequency, "Selecting most frequent students"):
        _students[student_id] = students[student_id]
    return _students


def select_n_most_frequent_students(source: str, target_prefix: str, ku_dict_path: str, n=None):
    """None in n means select all students"""
    n_list = as_list(n)
    students = _read(source, ku_dict_path)
    frequency = _frequency(students)
    for _n in n_list:
        _write(get_n_most_frequent_students(students, _n, frequency), target_prefix + str(_n) + ".csv")


def as_list(n):
    """Convert n to a list if it's not already a list"""
    return n if isinstance(n, list) else [n]