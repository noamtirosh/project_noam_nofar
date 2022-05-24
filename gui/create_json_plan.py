import json
#Opening JSON file
exercises_dict = {
    'exercises': [
        {
            'name': 'bridge',
            'num_of_reps': 20,
            'num_of_sets': 4,
            'break_time': 20,
            'count_model_path': r".\models\count_model.pth",
            'down_model_path': r".\models\down.pth",
            'err_message_sounds': [r"good_job1.mp3", r"hands_fix.mp3", r"neck_fix.mp3", r"hip_fix.mp3"],
            'open_message': r"open_message.mp3",
            'start_state_error_map': [0, 1],
            'middle_state_error_map': [0],
            'end_state_error_map': [0],

        }
    ],
    'plane_csv': r'home_exercise.csv'
}
json_string = json.dumps(exercises_dict)
with open('exercise_plan.json', 'w') as outfile:
    json.dump(json_string, outfile)
