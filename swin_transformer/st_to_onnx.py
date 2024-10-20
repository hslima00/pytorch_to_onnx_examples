import torch
import torch.onnx
import torch.nn as nn
from torch.nn import functional as F
import timm
from st_train import create_swin_transformer, NUM_CLASSES

def convert_pth_to_onnx(pth_path, onnx_path, input_shape):
    # Load the checkpoint
    checkpoint = torch.load(pth_path, map_location=torch.device("cpu"))

    # Create a new model instance with the correct number of classes
    model = create_swin_transformer(NUM_CLASSES)

    # Load the state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(input_shape)

    # Export the model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        training=torch.onnx.TrainingMode.EVAL,
        verbose=False,
    )

    print(f"Model converted to ONNX and saved at {onnx_path}")

if __name__ == "__main__":
    pth_path = "best_swin_transformer_model.pth"  # Path to your .pth file
    onnx_path = "swin_transformer_model.onnx"  # Desired path for the output .onnx file
    input_shape = (1, 3, 224, 224)  # Adjust based on your model's input shape (batch_size, channels, height, width)

    convert_pth_to_onnx(pth_path, onnx_path, input_shape)
