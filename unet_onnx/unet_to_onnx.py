import torch
import torch.onnx
import torch.nn as nn
from unet_train import (
    UNET,
    NUM_CLASSES,
)  # Import the UNET class and NUM_CLASSES from your training script


def convert_pth_to_onnx(pth_path, onnx_path, input_shape):
    # Load the checkpoint
    checkpoint = torch.load(
        pth_path, map_location=torch.device("cpu"), weights_only=True
    )

    # Create a new model instance with the correct number of classes
    model = UNET(in_channels=3, classes=NUM_CLASSES)

    # Load the state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # Check if the number of classes in the loaded model matches NUM_CLASSES
    if model.final_conv.out_channels != NUM_CLASSES:
        print(
            f"Adjusting final layer from {model.final_conv.out_channels} to {NUM_CLASSES} classes"
        )
        model.final_conv = nn.Conv2d(64, NUM_CLASSES, kernel_size=1)

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
    pth_path = "final_unet_model.pth"  # Path to your .pth file
    onnx_path = "unet_model.onnx"  # Desired path for the output .onnx file
    input_shape = (
        1,
        3,
        256,
        256,
    )  # Adjust based on your model's input shape (batch_size, channels, height, width)

    convert_pth_to_onnx(pth_path, onnx_path, input_shape)
