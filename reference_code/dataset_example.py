import yaml
from transformers import BertTokenizer

from dataset import Chalearn_MultiModalDataset
from dataset.transform import Chalearn_VideoTransform

config_path = "./task/baseline/config/base.yaml"
cfg = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

video_transform = Chalearn_VideoTransform()
train_dataset = Chalearn_MultiModalDataset(
    video_dir=cfg["datamodule"]["video_dir"],
    audio_dir=cfg["datamodule"]["audio_dir"],
    text_dir=cfg["datamodule"]["text_dir"],
    annotation_dir=cfg["datamodule"]["label_dir"]["train"],
    tokenizer=bert_tokenizer,
    target_list=cfg["datamodule"]["target_list"],
    text_max_len=cfg["datamodule"]["text_max_len"],
    frame_count=cfg["datamodule"]["frame_count"],
    seed=cfg["datamodule"]["seed"],
    video_transform=video_transform,
)

x, y = train_dataset.__getitem__(1)
video = x["video"]
audio = x["audio"]
text = x["text"]
print(video.shape)
print(audio.shape)
print(text[0].shape, text[1].shape, text[2].shape)
