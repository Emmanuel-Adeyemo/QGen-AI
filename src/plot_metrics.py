import os
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def plot_deep_eval_metrics():
    # dir setup
    ROOT = Path(__file__).resolve().parent.parent
    CSV_PATH = ROOT / 'output/metrics/final_deepeval_evaluation_metrics.csv'
    PLOT_DIR = ROOT / 'output/plots'
    PLOT_PATH = PLOT_DIR / 'deepeval_evaluation_metrics.png'

    # get metrics
    dta = pd.read_csv(CSV_PATH)

    metrics = dta['Metric']
    scores = dta['Average Score']
    pass_rates = dta['Pass Rate']


    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['text.color'] = '#1E293B'  # dark slate text
    plt.rcParams['axes.labelcolor'] = '#1E293B'

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)  # Ultra high-density resolution

    # Choose an intentional, professional single color accent (Slate Blue)
    bar_color = '#475569'
    bars = ax.barh(metrics, scores, color=bar_color, height=0.55, edgecolor='none', zorder=3)

    # 3. Clean up chart junk (Maximize data-ink ratio)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')  # Light gray muted y-axis spine
    ax.tick_params(axis='y', colors='#1E293B', labelsize=11, length=0)  # Hide ticks, keep text
    ax.tick_params(axis='x', colors='#64748B', labelsize=10)

    # Set strict bounding box for a RAG performance index (0.0 to 1.0)
    ax.set_xlim(0.0, 1.1)
    ax.set_xlabel('Empirical Performance Metric Score', fontsize=11, labelpad=12, fontweight='bold')

    # 4. Add subtle background grid lines only along the X axis
    ax.grid(axis='x', linestyle='--', alpha=0.4, color='#94A3B8', zorder=0)

    # 5. Smart Data Labeling: Inline scores + pass rates
    for bar, score, pass_rate in zip(bars, scores, pass_rates):
        width = bar.get_width()
        # Explicitly position the text payload immediately to the right of the bar
        ax.text(
            width + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f} ({pass_rate} Pass Rate)",
            va='center',
            ha='left',
            fontsize=10,
            fontweight='bold',
            color='#334155'
        )

    # 6. Apply absolute clean margins and output
    plt.title("QGen AI: Quantitative Genetics RAG Evaluation Index", fontsize=13, fontweight='bold', pad=18,
              loc='left', color='#0F172A')
    plt.tight_layout()

    # save
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(PLOT_PATH, bbox_inches='tight', transparent=True)
    print(f"Portfolio chart generated and exported successfully to {PLOT_DIR}")
    # plt.show()

if __name__ == '__main__':
    plot_deep_eval_metrics()