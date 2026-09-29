import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
from IPython.display import Markdown, display
from ipywidgets import Dropdown, interact
from tqdm.auto import tqdm
from transformers import pipeline

pd.set_option("display.max_columns", 160)
sns.set_theme(style = "whitegrid")
print("PyTorch:", torch.__version__)
print("CUDA-GPU tilgængelig:", torch.cuda.is_available())