import json
import pandas as pd
import numpy as np

from plot_util import set_plt_font_size
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.ticker import FuncFormatter

def draw_cost_breakdown_plot(df, file_path):
    fig_size = (12, 8)
    ax = df.plot(kind='bar', stacked=True, figsize=fig_size)
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    # plt.title('Stacked Bar Plot of Execution Time by Sequence Length and Cost Type')
    plt.legend()

    seq_lens = np.unique(np.array(list(df.index.str.split(' ')))[:,0].astype(int))
    parallelism_type_labels = np.array(list(df.index.str.split(' ')))[:,1].tolist()

    print(seq_lens)
    tick_labels = []
    tick_positions = []
    current_pos = 0
    for seq_len in seq_lens:
        indices = [index for index in df.index if index.startswith(str(seq_len))]
        n = len(indices)
        # Only add the sequence length label at the middle position of its group
        middle_pos = current_pos + n // 2
        tick_positions.append(middle_pos)
        tick_labels.append(seq_len)
        current_pos += n

    plt.xticks(tick_positions, tick_labels)  # Apply custom tick positions and labels

    # Add text on top of bar
    for rect, label in zip(ax.patches, parallelism_type_labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
        )

    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')

def get_df_from_dic(data):
    rows = []
    for seq_len, par_types in data.items():
        for par_type, costs in par_types.items():
            for cost_type, execution_time in costs.items():
                rows.append({'Sequence Length': seq_len, 'Parallelism Type': par_type, 'Cost Type': cost_type,
                             'Execution Time': execution_time})

    df = pd.DataFrame(rows)

    # Pivot the DataFrame to make it suitable for a stacked bar plot
    # df = df.pivot_table(index=['Sequence Length', 'Parallelism Type'], columns='Cost Type',
    #                           values='Execution Time', aggfunc='sum')
    df['Group'] = df['Sequence Length'] + ' ' + df['Parallelism Type']
    df = df.pivot_table(index='Group', columns='Cost Type', values='Execution Time', aggfunc='sum')

    # Step 3: Sort the DataFrame based on these numeric values
    # Create a temporary column for sorting, if you don't want to modify the original index
    numeric_index = np.array(list(df.index.str.split(' ')))[:,0].astype(int)
    df['SortKey'] = numeric_index
    df = df.sort_values(by='SortKey').drop(columns=['SortKey'])

    return df


if __name__ == "__main__":
    set_plt_font_size()
    result_path = "/home/byungsoj/eval_results/cost-breakdown.json"
    with open(result_path, 'r') as file:
        result_dic = json.load(file)

    df = get_df_from_dic(result_dic)
    out_file_path = f"/home/byungsoj/eval_results/cost_breakdown.pdf"
    draw_cost_breakdown_plot(df, out_file_path)
