import json

# Opening JSON file
exercises_dict = {
    'exercises': [
        {
            'name': 'bridge',
            'num_of_reps': 10,
            'num_of_sets': 4,
            'break_time': 15,
            'count_model_path': [r".\models\conv_count_with_err_right.pth",r".\models\conv_count_with_err_left.pth"],
            'start_state_model_path': [r".\models\conv_wight_down_right.pth",
                                       r".\models\conv_wight_down_left.pth"],
            'middle_state_model_path': [],
            'end_state_model_path': [],
            'err_message_sounds': [r".\error_sounds\ידיים.mp3", r".\error_sounds\כפות רגליים.mp3",
                                   r".\error_sounds\להרים ראש.mp3", r".\error_sounds\מצלמה.mp3"
                                   ],
            'open_message': r"open_message.mp3",
            'start_state_error_map': [0, 1, 2, 3],
            'middle_state_error_map': [0, 1, 2, 3],
            'end_state_error_map': [0, 1, 2, 3],
            'is_from_side': True

        }
    ],
    'plane_csv': r'home_exercise.csv'
}
json_string = json.dumps(exercises_dict)
with open('exercise_plan.json', 'w') as outfile:
    json.dump(json_string, outfile)
