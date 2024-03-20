import torch
import numpy as np

# interpretability method (you may wish to change)
from pytorchgradcam.pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, LayerCAM, FullGrad
# from pytorchgradcam.pytorch_grad_cam.utils import model_targets


def apply_gradcam(model: torch.nn.Module, target_layer, input_batch: torch.Tensor, method='HiResCam'):

    # using targets=None so that every image is calculated in relation to the highest-predicted class
    # can specify a particular class with the below (works badly)
    # targets=[MultiLabelBinaryOutputTarget(output_index=1, category=location)]
    # targets = [model_targets.BinaryClassifierOutputTarget(category=None)]
    # targets = [model_targets.ClassifierOutputTarget(category=1)]
    # targets = [model_targets.ClassifierOutputSoftmaxTarget(category=1)]


    cam_methods = {
        'GradCAM': GradCAM,
        'HiResCam': HiResCAM,
        'ScoreCAM': ScoreCAM,
        'GradCAMPlusPlus': GradCAMPlusPlus,
        'AblationCAM': AblationCAM,
        'XGradCAM': XGradCAM,
        'EigenCAM': EigenCAM,
        'LayerCAM': LayerCAM,
        'FullGrad': FullGrad
    }
    cam_method = cam_methods[method]

    cam = cam_method(model=model, target_layers=target_layer)

    output = [cam(input_tensor=torch.from_numpy(np.expand_dims(im, 0)), targets=None) for im in input_batch]

    return np.array(output)

