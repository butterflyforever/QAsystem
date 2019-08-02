import numpy as np
import difflib

from configs import *

s_list = []
g_list = []
with open(gif_csv, encoding='utf-8') as f:
    for line in f:
        c = line.strip().split('\t')
        if len(c) == 3:
            s_list.append(c[0])
            g_list.append(c[2])

ratios = np.zeros(len(s_list))
def get_gif(query, sim_thres=0.5, k=5):
    for i, s in enumerate(s_list):
        ratios[i] = difflib.SequenceMatcher(None, query, s).quick_ratio()

    selected = [g_list[i] for i in np.argpartition(ratios, -k)[-k:] if ratios[i] > sim_thres]
    if len(selected):
        selected = np.random.choice(selected)
        return selected
    else:
        return None