import torch
import torch.onnx
import torch.nn as nn
from torchvision import models
from dlv3_train import create_deeplabv3, NUM_CLASSES  # Import the create_deeplabv3 function and NUM_CLASSES from your training script

def convert_pth_to_onnx(pth_path, onnx_path, input_shape):
    # Load the checkpoint
    checkpoint = torch.load(pth_path, map_location=torch.device("cpu"))

    # Create a new model instance with the correct number of classes
    model = create_deeplabv3(NUM_CLASSES)

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
    pth_path = "final_deeplabv3_model.pth"  # Path to your .pth file
    onnx_path = "deeplabv3_model.onnx"  # Desired path for the output .onnx file
    input_shape = (1, 3, 256, 256)  # Adjust based on your model's input shape (batch_size, channels, height, width)

    convert_pth_to_onnx(pth_path, onnx_path, input_shape)
