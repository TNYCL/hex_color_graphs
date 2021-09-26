from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog as fd

root = tk.Tk()
root.withdraw()
filepath = fd.askopenfilename()

def rgb_to_hex(rgb_color, hex_color="#"):
    for i in rgb_color:
        i = int(i)
        hex_color += ("{:02x}".format(i))
    return hex_color

def prep_image(raw_image):
    modified_img = cv2.resize(raw_image, (600, 600), interpolation=cv2.INTER_AREA)
    modified_img = modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    return modified_img

def color_analysis(img):
    clf = KMeans(n_clusters=15)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    plt.figure(figsize=(12, 8))
    plt.pie(counts.values(), labels=hex_colors)
    plt.savefig("color_analysis_report.png")

image = cv2.imread(filepath)
modified_img = prep_image(image)
color_analysis(modified_img)