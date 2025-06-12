from openvino import Core
import numpy as np
import audio_functions as af
import os

# Load the IR model exported from ONNX
MODEL_PATH = "models\OpenVINO\WaveletModel-64k-2048.weights.xml"

ie = Core()
model = ie.read_model(model=MODEL_PATH)
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)

def preprocess_audio(file):
    # Read and preprocess audio file as in the original repo
    data = af.read_audiofile(file)
    # Padding and reshaping (example, adapt as needed)
    data_len = len(data)
    total_padsize = 2048
    if data_len % total_padsize > 0:
        data_len = data_len // total_padsize * total_padsize + total_padsize
    pad_size = data_len - len(data)
    zeros = np.zeros((pad_size, data.shape[1]), dtype=data.dtype, order="C")
    data = np.append(data, zeros, axis=0)
    # Reshape for model input
    data = np.reshape(data, (-1, 2048, 2))
    return data

def postprocess_output(output):
    # Flatten output if needed and return
    # (adapt this to match your specific output-to-audio logic)
    output = np.reshape(output, (-1, 2))
    return output

def main():
    print("Enter the path to the audio file you want to infer on:")
    audio_file = input().strip()
    if not os.path.isfile(audio_file):
        print(f"File '{audio_file}' does not exist.")
        return

    print(f"Processing {audio_file} ...")
    inputs = preprocess_audio(audio_file)
    # Infer in batches (if needed)
    outputs = []
    for batch in inputs:
        batch = batch[np.newaxis, ...]  # Add batch dimension
        out = compiled_model([batch])[compiled_model.output(0)]
        outputs.append(out[0])
    output = np.vstack(outputs)
    output_audio = postprocess_output(output)

    # Save output
    out_fname = os.path.splitext(audio_file)[0] + "_deloss.wav"
    af.write_audiofile_wav(output_audio, out_fname)
    print(f"Inference complete. Output saved as {out_fname}")

if __name__ == "__main__":
    main()