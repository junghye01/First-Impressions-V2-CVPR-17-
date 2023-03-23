import pandas as pd
from tqdm import tqdm
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)  # FutureWarning 제거

columns = ["video_name", "youtube_id", "transcription"]
df = pd.DataFrame(columns=columns)

pkl_list = ["transcription_training.pkl", "transcription_validation.pkl", "transcription_test.pkl"]
for pkl_filename in pkl_list:
    pkl = pd.read_pickle("./scripts/" + pkl_filename)
    print(pkl_filename)
    for video_name in tqdm(pkl.keys()):
        if video_name == "iYVJt41_q7M.002.mp4":
            pkl[
                "iYVJt41_q7M.002.mp4"
            ] = "take care. Thank you for showing so much love accross this channel subscribe as soon. I can't wait for giveaway and that happens as well. just thank you so much. Generally from the bottim. I had a lovely show on this channel. You guys take care and I will see you in the next video."
        if video_name == "4LZJvOecyM8.005.mp4":
            pkl[
                "4LZJvOecyM8.005.mp4"
            ] = "Interview with a vampire Queen of the Dammed. And that's all, I'm gonna leave for right now. Those are like big made ones. What did it..."
        if video_name == "YC3X1DcnUrk.000.mp4":
            pkl[
                "YC3X1DcnUrk.000.mp4"
            ] = "And spot, especially probably Princess Mononoke is just the most beautiful movie. It's just yeah. It's one of our favorite films in all categories. Not just in the animated sort of category Other Japanese animated films..."
        if video_name == "ztyBhnjtrz0.000.mp4":
            pkl[
                "ztyBhnjtrz0.000.mp4"
            ] = "Hey asks why are you so generous? Well I like to give things away and it's a nice thing to do even though I don't have a lot of money and I haven't enough money."
        if video_name == "JTmq4k4uQCY.003.mp4":
            pkl[
                "JTmq4k4uQCY.003.mp4"
            ] = "Building building and playing fun and the regular blocks are the best ones to give anything with them. Everyone knows all that about like already and Minecraft is that but you get to actually move around and play in this three world. I wouldn't like throw them into survival model..."
        if video_name == "KRo-x2uoHUg.003.mp4":
            pkl["KRo-x2uoHUg.003.mp4"] = "[Silence 00:00:18]"
        if video_name == "HhC2cGFFZeY.000.mp4":
            pkl[
                "HhC2cGFFZeY.000.mp4"
            ] = "Being 200 pounds would look monstrous like literally what feel like huge feel like a latch griffin? I think he's got some ability like bone density wise and things about similar height and he looks huge and he's only 170 pounds. I am about..."
        if video_name == "cRDYrvxRJ6U.001.mp4":
            pkl[
                "cRDYrvxRJ6U.001.mp4"
            ] = "And then what relationships they want to have they get really clear and really focused. They live a life. That's very purposeful to them. I absolutely love giving this my clients. It's so so so much fun. So that's kind of what you can expect working with a light coach..."
        if video_name == "JmAQlC-FEV8.000.mp4":
            pkl["JmAQlC-FEV8.000.mp4"] = "[Silence 00:00:18]"
        if video_name == "_plk5k7PBEg.004.mp4":
            pkl[
                "_plk5k7PBEg.004.mp4"
            ] = "I know in a video two videos ago, whatever was where I talked about being a normal tech youtuber isn't enough anymore. I want to make some things clear. I'm not saying I'm stopping all kinds of technology videos. I'll still do videos related to technology and I'll still do..."
        if video_name == "L-rmZZP_wj8.005.mp4":
            pkl["L-rmZZP_wj8.005.mp4"] = "[Playing music 00:00:18]"

        youtube_id = video_name[:-8]
        transcription = pkl[video_name]
        item = {"video_name": video_name, "youtube_id": youtube_id, "transcription": transcription}
        df = df.append(item, ignore_index=True)

df.to_csv("./scripts/transcription.csv", index=False)
