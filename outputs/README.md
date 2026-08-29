model_int8.tflitte

def convert_to_c_array(tflite_path, output_path, var_name="model_data"):
    with open(tflite_path, 'rb') as f:
        data = f.read()
    with open(output_path, 'w') as f:
        f.write(f"const unsigned char {var_name}[] = {{\n")
        for i, byte in enumerate(data):
            f.write(f"0x{byte:02x}, ")
            if (i+1) % 12 == 0:
                f.write("\n")
        f.write("\n};\n")
        f.write(f"const unsigned int {var_name}_len = {len(data)};\n")

convert_to_c_array('model_int8.tflite', 'model_data.h')

<img width="542" height="122" alt="image" src="https://github.com/user-attachments/assets/cd4f323e-b8d7-4cea-9480-a59ecc6535ef" />

