###
# Author: Kai Li
# Date: 2021-06-17 23:08:32
# LastEditors: Please set LastEditors
# LastEditTime: 2022-05-26 18:06:22
###
import torch
import torch.nn as nn

#from huggingface_hub import PyTorchModelHubMixin


def _unsqueeze_to_3d(x):
    """Normalize shape of `x` to [batch, n_chan, time]."""
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        return x


def pad_to_appropriate_length(x, lcm):
    values_to_pad = int(x.shape[-1]) % lcm
    if values_to_pad:
        appropriate_shape = x.shape
        padded_x = torch.zeros(
            list(appropriate_shape[:-1])
            + [appropriate_shape[-1] + lcm - values_to_pad],
            dtype=torch.float32,
        ).to(x.device)
        padded_x[..., : x.shape[-1]] = x
        return padded_x
    return x


class BaseModel(nn.Module):
    def __init__(self, sample_rate, in_chan=1):
        super().__init__()
        self._sample_rate = sample_rate
        self._in_chan = in_chan

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def sample_rate(self,):
        return self._sample_rate

    @staticmethod
    def load_state_dict_in_audio(model, pretrained_dict):
        model_dict = model.state_dict()
        update_dict = {}
        for k, v in pretrained_dict.items():
            if "audio_model" in k:
                update_dict[k[12:]] = v
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)
        return model

    @staticmethod
    def from_pretrain(pretrained_model_conf_or_path, *args, **kwargs):
        from . import get

        conf = torch.load(
            pretrained_model_conf_or_path, map_location="cpu"
        )  # Attempt to find the model and instantiate it.

        #print(conf["state_dict"])

        print(conf['infos'].keys())

        model_class = get(conf["model_name"])
        # model_class = get("Conv_TasNet")
        model = model_class(*args, **kwargs)
        model.load_state_dict(conf["state_dict"])
        return model

    def serialize(self):
        import pytorch_lightning as pl  # Not used in torch.hub

        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=self.get_model_args(),
        )
        # Additional infos
        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__, pytorch_lightning_version=pl.__version__,
        )
        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        """Should return args to re-instantiate the class."""
        raise NotImplementedError
