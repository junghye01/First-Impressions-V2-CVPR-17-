import yaml
from datamodule import Chalearn_BaselineLitDataModule
from network import MultiModalBaseModel, MultiModalEncoder, MultiModalRegressor
from module import Chalearn_BaselineLitModule
import lightning.pytorch as pl

config_path = "./task/baseline/config/base.yaml"
cfg = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

lit_data_module = Chalearn_BaselineLitDataModule(
    video_dir=cfg["datamodule"]["video_dir"],
    audio_dir=cfg["datamodule"]["audio_dir"],
    text_dir=cfg["datamodule"]["text_dir"],
    label_dir=cfg["datamodule"]["label_dir"],
    target_list=cfg["datamodule"]["target_list"],
    frame_count=cfg["datamodule"]["frame_count"],
    seed=cfg["datamodule"]["seed"],
    num_workers=cfg["datamodule"]["num_workers"],
    batch_size=cfg["datamodule"]["batch_size"],
)

multimodal_encoder = MultiModalEncoder(
    video_channels=cfg["module"]["encoder"]["video_channels"],
    video_input_size=cfg["module"]["encoder"]["video_input_size"],
    video_hidden_size=cfg["module"]["encoder"]["video_hidden_size"],
    video_num_layers=cfg["module"]["encoder"]["video_num_layers"],
    audio_input_size=cfg["module"]["encoder"]["audio_input_size"],
    audio_hidden_size=cfg["module"]["encoder"]["audio_hidden_size"],
    audio_num_layers=cfg["module"]["encoder"]["audio_num_layers"],
    text_input_size=cfg["module"]["encoder"]["text_input_size"],
    text_hidden_size=cfg["module"]["encoder"]["text_hidden_size"],
)

regressor = MultiModalRegressor(
    num_class=cfg["module"]["regressor"]["num_class"],
    video_hidden_size=cfg["module"]["encoder"]["video_hidden_size"],
    audio_hidden_size=cfg["module"]["encoder"]["audio_hidden_size"],
    text_hidden_size=cfg["module"]["encoder"]["text_hidden_size"],
    fusion_hidden_size=cfg["module"]["regressor"]["fusion_hidden_size"],
)
multimodal_model = MultiModalBaseModel(multimodal_encoder, regressor)

lit_module = Chalearn_BaselineLitModule(multimodal_model)

trainer = pl.Trainer(devices=[0], limit_train_batches=100, max_epochs=1)
trainer.fit(model=lit_module, train_dataloaders=lit_data_module)
