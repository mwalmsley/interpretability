import shap
import numpy as np
import matplotlib.pyplot as plt

def nhwc_to_nchw(x: np.ndarray) -> torch.Tensor:
    if x.ndim == 4:
        return torch.from_numpy(x).permute(0, 3, 1, 2)
    elif x.ndim == 3:
        return torch.from_numpy(x).permute(2, 0, 1)
    else:
        raise ValueError(f"Unsupported number of dimensions: {x.ndim}")

def nchw_to_nhwc(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 4:
        return x.permute(0, 2, 3, 1).cpu().numpy()
    elif x.ndim == 3:
        return x.permute(1, 2, 0).cpu().numpy()
    else:
        raise ValueError(f"Unsupported number of dimensions: {x.ndim}")

def predict(img: np.ndarray) -> torch.Tensor:
    img = torch.Tensor(img)
    img = nhwc_to_nchw(img)
    img = img.to(device)
    output = model(img)
    return output


def calculate_shap_values(input_data: np.ndarray, model_predict_function, masker, num_samples: int = 5000,
                          batch_size: int = 50):
    explainer = shap.Explainer(model_predict_function, masker, output_names=["class 0", "class 1"])
    shap_values = explainer(input_data, max_evals=num_samples, batch_size=batch_size)
    return shap_values


def visualize_shap_values(shap_values, input_data: np.ndarray):
    shap_values_to_plot = [shap_values.values[:, :, :, :, class_index] for class_index in
                           range(shap_values.values.shape[-1])]
    labels_to_plot = np.array([['not_ring', 'ring'] for _ in range(input_data.shape[0])])

    shap.image_plot(
        shap_values=shap_values_to_plot,
        pixel_values=input_data,
        labels=labels_to_plot
    )


input_batch_np = np.moveaxis(img_tensor.numpy(), 1, 3)

image_shape = input_batch_np[0].shape
masker_blur = shap.maskers.Image("blur(15,15)", image_shape)
shap_values = calculate_shap_values(input_batch_np, predict, masker_blur)


visualize_shap_values(shap_values, input_batch_np)
