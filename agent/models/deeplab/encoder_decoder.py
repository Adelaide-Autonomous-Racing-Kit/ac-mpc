import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

    """
        Retrieves intermediate representation's number of channels
    """

    @property
    def _encoder_channels(self):
        currentLevel, chnl_sizes, tmpParam = "_level1", [], None
        # Get the paramters from the model and their names
        for name, param in self._encoder.named_parameters():
            tmpLevel = name.split(".")[0]
            # If it is the last layer of a level, add its name and size
            if not tmpLevel == currentLevel:
                chnl_sizes.append((currentLevel, tmpParam.size()[0]))
                currentLevel = tmpLevel
            tmpParam = param

        chnl_sizes.append((tmpLevel, tmpParam.size()[0]))
        return chnl_sizes
