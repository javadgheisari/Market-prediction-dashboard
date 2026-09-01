import matplotlib.pyplot as plt
import pickle
from easygui import fileopenbox

address = fileopenbox(title="choose pkl file")

# Load the plot from the pickle file
with open(address, 'rb') as f:
    fig = pickle.load(f)

# Display the plot
plt.show()