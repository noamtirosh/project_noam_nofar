import json
#Opening JSON file
exercises_dict = {
    'exercises': [
        {
            'name': 'bridge',
            'num_of_reps': 20,
            'num_of_sets': 4,
            'break_time': 20,
            'count_model_path': r"C:\git_repos\project_noam_nofar\models\work very good\model_angle_vec_20.4.pth",
            'down_model_path': r"C:\git_repos\project_noam_nofar\models\down_model.pth",
            'no_mistake': r"good_job1.mp3",
            'error_type1': r"hands_fix.mp3",
            'error_type2': r"neck_fix.mp3",
            'error_type3': r"hip_fix.mp3",
            'open_message': r"open_message.mp3",
        }
    ],
    'plane_csv': r'home_exercise.csv'
}
json_string = json.dumps(exercises_dict)
with open('exercise_plan_18-5.json', 'w') as outfile:
    json.dump(json_string, outfile)
