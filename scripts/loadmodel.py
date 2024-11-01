# Function to load model

from zoobot.pytorch.training import finetune
from zoobot.pytorch.training import representations

# Note that JWST is loaded separately as a ckpt. Otherwise import from huggingface

def load_model(dataset: str, jwst_loc: str = None):
    """Loads model according to dataset name (Decals10, GZ_Rings, JWST, etc.).
    For JWST, takes file path as argument to load locally.
    """

    if dataset == 'jwst':
        base_model = representations.ZoobotEncoder.load_from_name('hf_hub:mwalmsley/zoobot-encoder-convnext_nano')
        if jwst_loc is None:
            raise ValueError("No filepath provided for JWST model.")
        model = finetune.FinetuneableZoobotClassifier.load_from_checkpoint(jwst_loc, encoder=base_model.encoder)

    else:
        model_path = 'mwalmsley/zoobot-finetuned-' + dataset
        model = finetune.FinetuneableZoobotClassifier.load_from_name(model_path)

    return model