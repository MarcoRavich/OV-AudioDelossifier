import os
import sys
import tensorflow as tf
import tf2onnx
import audio_models as models
import audio_config as cfg

MODELS_DIR = "models"

def list_model_files():
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".h5") or f.endswith(".weights.h5")]
    for i, fname in enumerate(files):
        print(f"{i+1}: {fname}")
    return files

def main():
    files = list_model_files()
    if not files:
        print(f"No .h5 or .weights.h5 files found in '{MODELS_DIR}'")
        return

    print("Enter the number of the model you want to convert to ONNX:")
    idx = input().strip()
    try:
        idx = int(idx) - 1
        model_file = files[idx]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    # Guess bitrate from filename (e.g., 'WaveletModel-64k-2048.weights.h5' -> '64k')
    bitrate = None
    for rate in ["64k", "96k", "128k", "192k", "256k", "320k"]:
        if rate in model_file:
            bitrate = rate
            break

    if bitrate is None:
        print("Could not infer bitrate from filename. Please enter bitrate (e.g., 64k):")
        bitrate = input().strip()

    print(f"Building model and loading weights from {model_file} (bitrate {bitrate})")

    model = models.build_model()
    models.load_weights(model, bitrate)

    spec = (tf.TensorSpec((None, cfg.InputSize, cfg.NumChannels), tf.float32, name="input"),)
    output_path = os.path.splitext(model_file)[0] + ".onnx"
    output_path = os.path.join(MODELS_DIR, output_path)

    print(f"Exporting to ONNX: {output_path}")
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
    print(f"Saved ONNX: {output_path}")

if __name__ == "__main__":
    # Make sure we can import repo modules regardless of run location
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()