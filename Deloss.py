from openvino import Core
import numpy as np
import audio_functions as af
import os
import re
import time

LOSSY_EXTENSIONS = {"mp3", "m4a", "aac", "ac3", "ogg", "opus", "wma"}
LOSSLESS_EXTENSIONS = {"wav", "flac", "ape", "alac", "wv", "aiff", "aif"}

def get_file_extension(filename):
    return os.path.splitext(filename)[1][1:].lower()

def select_device(ie):
    available_devices = ie.available_devices
    device_map = {'CPU': 'CPU', 'GPU': 'GPU', 'NPU': 'NPU'}
    devices_present = [d for d in device_map if any(dev.startswith(device_map[d]) for dev in available_devices)]
    print("\nAvailable devices for inference:")
    for i, d in enumerate(devices_present):
        print(f"{i+1}: {d}")
    print(f"{len(devices_present)+1}: AUTO (Automatic Device Selection)")
    print("Choose the device to run inference on (number):")
    while True:
        try:
            choice = int(input().strip())
            if 1 <= choice <= len(devices_present):
                return device_map[devices_present[choice-1]]
            elif choice == len(devices_present)+1:
                auto_devices = [device_map[d] for d in devices_present]
                return "AUTO:" + ",".join(auto_devices)
            else:
                print("Invalid choice. Please select a valid number.")
        except Exception:
            print("Invalid input. Please enter a number.")

def get_models_dir():
    print("Enter the path to the models directory (default: ./models/OpenVINO):")
    models_dir = input().strip()
    if not models_dir:
        models_dir = os.path.join("models", "OpenVINO")
    if not os.path.isdir(models_dir):
        print(f"Directory '{models_dir}' does not exist.")
        exit(1)
    return models_dir

def parse_bitrate_from_model(filename):
    match = re.search(r'(\d+)k', filename)
    if match:
        return int(match.group(1))
    return None

def find_models(models_dir):
    models = []
    for fname in os.listdir(models_dir):
        if fname.endswith(".xml"):
            bitrate = parse_bitrate_from_model(fname)
            if bitrate is not None:
                models.append((bitrate, os.path.join(models_dir, fname)))
    return dict(models)

def get_audio_bitrate(audio_file):
    if hasattr(af, 'get_audio_bitrate'):
        return af.get_audio_bitrate(audio_file)
    # fallback: quick samplerate-based guess
    try:
        sr, _ = af.read_audiofile_basic(audio_file)
        if sr >= 44000:
            return 320
        elif sr > 20000:
            return 128
        else:
            return 64
    except Exception:
        return None

def ask_user_confirm_bitrate(auto_bitrate):
    print(f"Detected lossy input bitrate: {auto_bitrate} kbps")
    print("Is this correct? Enter different kbps value or press Enter to confirm:")
    inp = input().strip()
    if inp == "":
        return auto_bitrate
    try:
        val = int(inp)
        if val > 0:
            return val
    except Exception:
        pass
    print("Invalid input, using detected bitrate.")
    return auto_bitrate

def choose_model_user(models_dict):
    print("Available models:")
    items = sorted(models_dict.items())
    for i, (br, path) in enumerate(items, 1):
        print(f"{i}: {os.path.basename(path)} - {br} kbps")
    print("Choose the model to use (number):")
    while True:
        try:
            choice = int(input().strip())
            if 1 <= choice <= len(items):
                return items[choice-1][1]
            else:
                print("Invalid choice. Please select a valid number.")
        except Exception:
            print("Invalid input. Please enter a number.")

def choose_model_auto(models_dict, bitrate):
    if bitrate in models_dict:
        return models_dict[bitrate]
    sorted_bitrates = sorted(models_dict.keys())
    candidates = [br for br in sorted_bitrates if br <= bitrate]
    if candidates:
        return models_dict[candidates[-1]]
    # fallback: ask user to choose
    print("Could not auto-select a model for this bitrate.")
    return choose_model_user(models_dict)

def preprocess_audio(file):
    data, _ = af.read_audiofile(file)
    data_len = len(data)
    total_padsize = 2048
    if data_len % total_padsize > 0:
        data_len = data_len // total_padsize * total_padsize + total_padsize
    pad_size = data_len - len(data)
    zeros = np.zeros((pad_size, data.shape[1]), dtype=data.dtype, order="C")
    data = np.append(data, zeros, axis=0)
    data = np.reshape(data, (-1, 2048, 2))
    return data

def postprocess_output(output):
    output = np.reshape(output, (-1, 2))
    return output.astype(np.float32)

def main():
    ie = Core()
    device = select_device(ie)
    models_dir = get_models_dir()
    models_dict = find_models(models_dir)

    if not models_dict:
        print("No valid model files found in the specified directory.")
        exit(1)

    print("Enter the path to the audio file you want to infer on:")
    audio_file = input().strip()
    if not os.path.isfile(audio_file):
        print(f"File '{audio_file}' does not exist.")
        return

    ext = get_file_extension(audio_file)
    lossy = ext in LOSSY_EXTENSIONS
    lossless = ext in LOSSLESS_EXTENSIONS

    if lossy:
        auto_bitrate = get_audio_bitrate(audio_file)
        bitrate = ask_user_confirm_bitrate(auto_bitrate)
        model_path = choose_model_auto(models_dict, bitrate)
    else:
        print("Lossless or unknown audio format detected - manual model selection required.")
        model_path = choose_model_user(models_dict)

    print(f"Using model: {model_path}")
    print(f"Using device: {device}")

    print(f"Processing {audio_file} ...")
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name=device)
    input_layer = compiled_model.input(0)

    inputs = preprocess_audio(audio_file)
    outputs = []

    start_time = time.time()
    for batch in inputs:
        batch = batch[np.newaxis, ...]
        out = compiled_model([batch])[compiled_model.output(0)]
        outputs.append(out[0])
    inference_time = time.time() - start_time

    output = np.vstack(outputs)
    output_audio = postprocess_output(output)

    out_fname = os.path.splitext(audio_file)[0] + "_deloss.wav"
    af.write_audiofile_wav(output_audio, out_fname, dtype=np.float32)
    print(f"Inference complete. Output saved as {out_fname} (32bit float PCM WAV)")
    print(f"Inference time: {inference_time:.2f} seconds")

if __name__ == "__main__":
    main()
