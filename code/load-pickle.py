import matplotlib.pyplot as plt
import pickle
from datetime import datetime

# print(type(datetime.now().strftime("%Y-%m-%d")))

# Load the plot from the pickle file
with open('plots/BTC/2023-11-17.pkl', 'rb') as f:
    fig = pickle.load(f)

# Display the plot
plt.show()