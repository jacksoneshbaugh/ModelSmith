"""
Creates a model config YAML file for use in ModelSmith.
"""

__author__ = 'Jackson Eshbaugh'
__version__ = '12/17/2024'

import tkinter as tk
from tkinter import ttk, filedialog
import yaml
from keras import layers as keras_layers

# Globals
data = {}
model = 'sequential'
layers = []
training = {}
random_seed = 0


def write_model_config():
    """
    Writes the model config YAML file based on the parameters set in the global variables.
    """
    config = {
        "data": data,
        "model": model,
        "layers": layers,
        "training": training,
        "random_seed": random_seed
    }

    file_path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML files", "*.yaml")])
    if file_path:
        with open(file_path, 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False, sort_keys=False)
        tk.messagebox.showinfo("Success", "Configuration saved successfully!")


def add_layer(layer_type, params, update_layer_display):
    """
    Adds a new layer to the layers list and refreshes the display.
    :param layer_type: The type of the layer (e.g., Dense, Input).
    :param params: Dictionary of parameters for the layer.
    :param update_layer_display: Function to refresh the displayed layers.
    """
    layer = {"type": layer_type}
    layer.update(params)
    layers.append(layer)
    print(f"Added layer: {layer}")
    update_layer_display()


def configure_data():
    """
    Opens a window to configure the data dictionary.
    """
    def save_data():
        data['type'] = type_var.get()
        data['path'] = path_entry.get()
        data['inputs'] = inputs_entry.get().split(',')
        data['outputs'] = outputs_entry.get().split(',')
        print(f"Data configuration: {data}")
        data_window.destroy()

    data_window = tk.Toplevel()
    data_window.title("Configure Data")

    tk.Label(data_window, text="Type:").grid(row=0, column=0, sticky='e')
    type_var = tk.StringVar(value=data.get('type', 'csv'))
    ttk.Entry(data_window, textvariable=type_var).grid(row=0, column=1)

    tk.Label(data_window, text="Path:").grid(row=1, column=0, sticky='e')
    path_entry = ttk.Entry(data_window)
    path_entry.insert(0, data.get('path', ''))
    path_entry.grid(row=1, column=1)

    tk.Label(data_window, text="Inputs (comma-separated):").grid(row=2, column=0, sticky='e')
    inputs_entry = ttk.Entry(data_window)
    inputs_entry.insert(0, ','.join(data.get('inputs', [])))
    inputs_entry.grid(row=2, column=1)

    tk.Label(data_window, text="Outputs (comma-separated):").grid(row=3, column=0, sticky='e')
    outputs_entry = ttk.Entry(data_window)
    outputs_entry.insert(0, ','.join(data.get('outputs', [])))
    outputs_entry.grid(row=3, column=1)

    ttk.Button(data_window, text="Save", command=save_data).grid(row=4, column=0, columnspan=2)


def configure_training():
    """
    Opens a window to configure the training dictionary.
    """
    def save_training():
        training['optimizer'] = optimizer_var.get()
        training['loss'] = loss_var.get()
        training['learning_rate'] = float(lr_entry.get())
        training['metrics'] = metrics_entry.get().split(',')
        training['batch_size'] = int(batch_size_entry.get())
        training['epochs'] = int(epochs_entry.get())
        training['validation_split'] = float(validation_split_entry.get())
        print(f"Training configuration: {training}")
        training_window.destroy()

    training_window = tk.Toplevel()
    training_window.title("Configure Training")

    tk.Label(training_window, text="Optimizer:").grid(row=0, column=0, sticky='e')
    optimizer_var = tk.StringVar(value=training.get('optimizer', 'adam'))
    ttk.Entry(training_window, textvariable=optimizer_var).grid(row=0, column=1)

    tk.Label(training_window, text="Loss:").grid(row=1, column=0, sticky='e')
    loss_var = tk.StringVar(value=training.get('loss', 'binary_crossentropy'))
    ttk.Entry(training_window, textvariable=loss_var).grid(row=1, column=1)

    tk.Label(training_window, text="Learning Rate:").grid(row=2, column=0, sticky='e')
    lr_entry = ttk.Entry(training_window)
    lr_entry.insert(0, training.get('learning_rate', 0.001))
    lr_entry.grid(row=2, column=1)

    tk.Label(training_window, text="Metrics (comma-separated):").grid(row=3, column=0, sticky='e')
    metrics_entry = ttk.Entry(training_window)
    metrics_entry.insert(0, ','.join(training.get('metrics', ['accuracy'])))
    metrics_entry.grid(row=3, column=1)

    tk.Label(training_window, text="Batch Size:").grid(row=4, column=0, sticky='e')
    batch_size_entry = ttk.Entry(training_window)
    batch_size_entry.insert(0, training.get('batch_size', 8))
    batch_size_entry.grid(row=4, column=1)

    tk.Label(training_window, text="Epochs:").grid(row=5, column=0, sticky='e')
    epochs_entry = ttk.Entry(training_window)
    epochs_entry.insert(0, training.get('epochs', 1000))
    epochs_entry.grid(row=5, column=1)

    tk.Label(training_window, text="Validation Split:").grid(row=6, column=0, sticky='e')
    validation_split_entry = ttk.Entry(training_window)
    validation_split_entry.insert(0, training.get('validation_split', 0.2))
    validation_split_entry.grid(row=6, column=1)

    ttk.Button(training_window, text="Save", command=save_training).grid(row=7, column=0, columnspan=2)


def configure_layers(update_layer_display):
    """
    Opens a window to add layers to the layers list with dynamic parameters.
    """
    def add_layer_to_list():
        layer_type = layer_type_var.get()
        params = {}
        for param, entry in param_entries.items():
            value = entry.get()
            try:
                value = eval(value)  # Convert to proper type if possible
            except:
                pass
            params[param] = value
        add_layer(layer_type, params, update_layer_display)
        layer_window.destroy()

    def add_param():
        param_name = param_name_entry.get()
        if param_name:
            row = len(param_entries) + 2
            tk.Label(param_frame, text=param_name).grid(row=row, column=0, sticky='e')
            entry = ttk.Entry(param_frame)
            entry.grid(row=row, column=1)
            param_entries[param_name] = entry
            param_name_entry.delete(0, tk.END)

    def remove_param():
        if param_entries:
            param_name, entry = param_entries.popitem()
            entry.grid_forget()
            print(f"Removed parameter: {param_name}")

    layer_window = tk.Toplevel()
    layer_window.title("Add Layer")

    tk.Label(layer_window, text="Layer Type:").grid(row=0, column=0, sticky='e')
    layer_type_var = tk.StringVar()
    layer_types = [cls.__name__ for cls in keras_layers.__dict__.values() if isinstance(cls, type)]
    ttk.Combobox(layer_window, textvariable=layer_type_var, values=layer_types).grid(row=0, column=1)

    param_frame = tk.Frame(layer_window)
    param_frame.grid(row=1, column=0, columnspan=2)

    tk.Label(param_frame, text="Add Parameter:").grid(row=0, column=0)
    param_name_entry = ttk.Entry(param_frame)
    param_name_entry.grid(row=0, column=1)
    ttk.Button(param_frame, text="Add", command=add_param).grid(row=0, column=2)
    ttk.Button(param_frame, text="Remove", command=remove_param).grid(row=0, column=3)

    param_entries = {}

    ttk.Button(layer_window, text="Add Layer", command=add_layer_to_list).grid(row=2, column=0, columnspan=2)


def configure_model():
    """
    Opens a popup to configure the model type.
    """
    def save_model():
        global model
        model = model_var.get()
        print(f"Model configuration set to: {model}")
        model_window.destroy()

    # Create a popup window
    model_window = tk.Toplevel()
    model_window.title("Configure Model")

    tk.Label(model_window, text="Select Model Type:").grid(row=0, column=0, padx=10, pady=10)

    # Dropdown for selecting model type
    model_var = tk.StringVar(value=model)  # Default to the current model value
    model_options = ["sequential"]  # Add other model types as needed
    ttk.Combobox(model_window, textvariable=model_var, values=model_options).grid(row=0, column=1, padx=10, pady=10)

    # Save button
    ttk.Button(model_window, text="Save", command=save_model).grid(row=1, column=0, columnspan=2, pady=10)


def main():
    def update_layer_display():
        for widget in layer_frame.winfo_children():
            widget.destroy()

        for idx, layer in enumerate(layers):
            layer_text = f"{idx + 1}: {layer['type']} ({', '.join(f'{k}={v}' for k, v in layer.items() if k != 'type')})"
            tk.Label(layer_frame, text=layer_text).pack(anchor='w')

    root = tk.Tk()
    root.title("ModelSmith Configurator")

    action_frame = tk.Frame(root)
    action_frame.pack(side=tk.LEFT, padx=10, pady=10)

    ttk.Button(action_frame, text="Configure Data", command=configure_data).pack(pady=5)
    ttk.Button(action_frame, text="Configure Training", command=configure_training).pack(pady=5)
    ttk.Button(action_frame, text="Configure Model", command=configure_model).pack(pady=5)
    ttk.Button(action_frame, text="Add Layer", command=lambda: configure_layers(update_layer_display)).pack(pady=5)
    ttk.Button(action_frame, text="Save Config", command=write_model_config).pack(pady=5)

    layer_frame = tk.Frame(root)
    layer_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    tk.Label(layer_frame, text="Layers:").pack(anchor='w')

    root.mainloop()


if __name__ == "__main__":
    main()