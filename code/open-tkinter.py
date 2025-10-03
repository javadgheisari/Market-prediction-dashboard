import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import pickle

def open_file():
    # Open file dialog to select the .pkl file
    filepath = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
    
    if filepath:
        try:
            # Load the plot from the pickle file
            with open(filepath, 'rb') as f:
                fig = pickle.load(f)

            # Display the plot
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {e}")


# Create the Tkinter application window
window = tk.Tk()
window.title("Plot Viewer")

# Create a label
label = tk.Label(window, text="Click the button to open a .pkl file", font=("Arial", 14))
label.pack(pady=20)

# Create a button to open the file
open_button = tk.Button(window, text="Open File", font=("Arial", 12), command=open_file)
open_button.pack(pady=10)

# Configure window appearance
window.geometry("400x200")
window.resizable(False, False)

# Start the Tkinter event loop
window.mainloop()