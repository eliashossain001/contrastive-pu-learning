import os
import matplotlib.pyplot as plt
from datetime import datetime

def init_output_dir(base="outputs"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_plot(fig, filename, output_dir):
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
