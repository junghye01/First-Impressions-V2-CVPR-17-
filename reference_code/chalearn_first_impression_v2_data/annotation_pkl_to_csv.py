import pandas as pd
from tqdm import tqdm
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)  # FutureWarning 제거


tmp_1 = pd.read_csv("./annotation/pkl_and_csv/eth_gender_annotations_dev.csv", sep=";")
tmp_2 = pd.read_csv("./annotation/pkl_and_csv/eth_gender_annotations_test.csv", sep=";")
eth_gender_df = pd.concat([tmp_1, tmp_2]).reset_index(drop=True)


def get_annotation_df():
    columns = [
        "video_name",
        "youtube_id",
        "ethnicity",
        "gender",
        "extraversion",
        "neuroticism",
        "agreeableness",
        "conscientiousness",
        "interview",
        "openness",
    ]
    return pd.DataFrame(columns=columns)


annotation_pkls = {
    "annotation_training.pkl": "annotation_train.csv",
    "annotation_validation.pkl": "annotation_validation.csv",
    "annotation_test.pkl": "annotation_test.csv",
}
for annotation_filename in annotation_pkls.keys():
    annotation = pd.read_pickle("./annotation/pkl_and_csv/" + annotation_filename)
    filenames = list(annotation["extraversion"].keys())
    print(annotation_filename)
    annotation_df = get_annotation_df()
    for video_name in tqdm(filenames):
        tmp = eth_gender_df[eth_gender_df["VideoName"] == video_name].reset_index(drop=True)
        video_name = tmp.loc[0]["VideoName"]
        youtube_id = tmp.loc[0]["YouTubeID"]
        ethnicity = tmp.loc[0]["Ethnicity"]
        ethnicity_label = None
        if ethnicity == 1:
            ethnicity = "Asian"
            ethnicity_label = 0
        elif ethnicity == 2:
            ethnicity = "Caucasian"
            ethnicity_label = 1
        elif ethnicity == 3:
            ethnicity = "African-American"
            ethnicity_label = 2
        gender = tmp.loc[0]["Gender"]
        gender_label = None
        if gender == 1:
            gender = "Male"
            gender_label = 0
        elif gender == 2:
            gender = "Female"
            gender_label = 1
        extraversion = annotation["extraversion"][video_name]
        neuroticism = annotation["neuroticism"][video_name]
        agreeableness = annotation["agreeableness"][video_name]
        conscientiousness = annotation["conscientiousness"][video_name]
        interview = annotation["interview"][video_name]
        openness = annotation["openness"][video_name]
        item = {
            "video_name": video_name,
            "youtube_id": youtube_id,
            "ethnicity": ethnicity,
            "ethnicity_label": ethnicity_label,
            "gender": gender,
            "gender_label": gender_label,
            "extraversion": extraversion,
            "neuroticism": neuroticism,
            "agreeableness": agreeableness,
            "conscientiousness": conscientiousness,
            "interview": interview,
            "openness": openness,
        }
        annotation_df = annotation_df.append(item, ignore_index=True)
    print("./annotation/original/" + annotation_pkls[annotation_filename])
    # annotation_df.to_csv("./annotation/original/" + annotation_pkls[annotation_filename])
    print(annotation_df)
