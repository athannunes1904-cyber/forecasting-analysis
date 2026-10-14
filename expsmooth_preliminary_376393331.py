import pandas as pd
import matplotlib.pyplot as plt
import os

data_path = os.path.join("..", "data_clean", "combined_clean.xlsx")
figures_path = os.path.join("..", "figures")

df = pd.read_excel(data_path)

df["date_excel"] = pd.to_datetime(df["date_excel"])
df = df.sort_values("date_excel")
df = df.set_index("date_excel")

series_list = ["gb7s", "k55o", "gmaa", "gmak"]

for col in series_list:
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[col])
    plt.title(col.upper() + " Time Plot")
    plt.xlabel("Date")
    plt.ylabel(col.upper())
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, col.upper() + "_time_plot.png"), dpi=300)
    plt.show()