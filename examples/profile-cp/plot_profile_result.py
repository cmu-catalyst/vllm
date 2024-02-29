import json
import pandas as pd
from plot_util import set_plt_font_size
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.ticker import FuncFormatter

def draw_cost_breakdown_plot(df, seq_lens, file_path):
    fig_size = (12, 8)
    df.plot(kind='bar', stacked=True, figsize=fig_size)
    plt.xlabel('Sequence Length and Parallelism Type')
    plt.ylabel('Time (ms)')
    # plt.title('Stacked Bar Plot of Execution Time by Sequence Length and Cost Type')
    plt.legend()

    print(df)

    tick_labels = []
    tick_positions = []
    current_pos = 0
    for seq_len in seq_lens:
        indices = [index for index in df.index if index.startswith(seq_len)]
        n = len(indices)
        # Only add the sequence length label at the middle position of its group
        middle_pos = current_pos + n // 2
        tick_positions.append(middle_pos)
        tick_labels.append(seq_len)
        current_pos += n

    plt.xticks(tick_positions, tick_labels)  # Apply custom tick positions and labels

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

    seq_lens = df['Sequence Length'].unique()
    df['Group'] = df['Sequence Length'] + ' ' + df['Parallelism Type']
    df = df.pivot_table(index='Group', columns='Cost Type', values='Execution Time', aggfunc='sum')

    return df, seq_lens


if __name__ == "__main__":
    set_plt_font_size()
    result_path = "/home/byungsoj/eval_results/cost-breakdown.json"
    with open(result_path, 'r') as file:
        result_dic = json.load(file)

    df, seq_lens = get_df_from_dic(result_dic)
    out_file_path = f"/home/byungsoj/eval_results/cost_breakdown.pdf"
    draw_cost_breakdown_plot(df, seq_lens, out_file_path)
