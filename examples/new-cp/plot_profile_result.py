import json
import pandas as pd
import numpy as np
import copy

from plot_util import set_plt_font_size
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.ticker import FuncFormatter

def to_k(value, pos=None):
    """Convert number to thousand (K) format."""
    return f'{value / 1000.0:.1f}K'

    """Convert number to thousand (K) format."""
def draw_cost_breakdown_plot(df, file_path, cost_type):
    fig_size = (12, 9)

    df_plot = df.pivot(index='Sequence Length', columns='Parallelism Type', values=cost_type)
    df_plot = df_plot/1000.0
    df_plot.index = df_plot.index.map(to_k)

    if cost_type == "comm":
        df_plot.plot.bar(rot=0, figsize=fig_size, color=['C2', 'C0'])
    else:
        df_plot.plot.bar(rot=0, figsize=fig_size, color=['C2', 'C1', 'C0'])

    # formatter = FuncFormatter(to_k)
    # plt.gca().yaxis.set_major_formatter(formatter)

    plt.xlabel("Sequence Length")
    plt.ylabel("Execution Time (ms)")
    # plt.title(f"Execution Time for {cost_type.capitalize()} Cost Type")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')

def get_df_from_dic(data):
    df = pd.DataFrame.from_dict({(i, j): data[i][j] for i in data.keys() for j in data[i].keys()}, orient='index')
    df.index.names = ['Sequence Length', 'Parallelism Type']
    df = df.reset_index()
    df['Parallelism Type'] = df['Parallelism Type'].str.upper()
    df['Sequence Length'] = df['Sequence Length'].astype(int)
    # print(df)
    return df


if __name__ == "__main__":
    set_plt_font_size()
    result_path = "/home/byungsoj/eval_results/final_result/round2/cost-breakdown.json"
    with open(result_path, 'r') as file:
        result_dic = json.load(file)

    df = get_df_from_dic(result_dic)

    # Iterate over cost types and create separate plots
    for cost_type in ["comp", "others", "comm"]:
        out_file_path = f"/home/byungsoj/eval_results/plots/cost_breakdown_{cost_type}.pdf"
        df_to_plot = copy.deepcopy(df)
        if cost_type == "comm":
            df_to_plot = df_to_plot[df_to_plot['Parallelism Type'] != 'DP']
        draw_cost_breakdown_plot(df_to_plot, out_file_path, cost_type)