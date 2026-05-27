"""Visualization utilities for experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

from src.models.automata.transitions import TransitionModel


def plot_confusion_matrix(y_true, y_pred, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_prob, title: str, output_path: Path) -> None:
    if len(np.unique(y_true)) < 2:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(y_true, y_prob, title: str, output_path: Path) -> None:
    if len(np.unique(y_true)) < 2:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_transition_heatmap(transition_model: TransitionModel, output_path: Path, title: str = "Transition Probabilities") -> None:
    states = sorted(transition_model.states)
    if not states:
        return
    matrix = np.zeros((len(states), len(states)))
    for i, src in enumerate(states):
        for j, dst in enumerate(states):
            matrix[i, j] = transition_model.transition_probability(src, dst)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=states, yticklabels=states, cmap="YlOrRd", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_state_diagram(transition_model: TransitionModel, output_path: Path, min_prob: float = 0.05) -> None:
    G = nx.DiGraph()
    for src in transition_model.states:
        G.add_node(src)
        for dst, _count in transition_model.counts.get(src, {}).items():
            prob = transition_model.transition_probability(src, dst)
            if prob >= min_prob:
                G.add_edge(src, dst, weight=prob, label=f"{prob:.2f}")

    if len(G.nodes) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#4C72B0", node_size=800)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]
    nx.draw_networkx_edges(G, pos, ax=ax, width=[w * 3 for w in weights], arrows=True)
    ax.set_title("Automata State Diagram")
    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_parameter_sensitivity(records: list[dict], param_name: str, output_path: Path) -> None:
    if not records:
        return
    values = [r[param_name] for r in records]
    f1_scores = [r.get("f1_mean", r.get("f1", 0)) for r in records]
    state_counts = [r.get("state_count", 0) for r in records]
    densities = [r.get("transition_density", 0) for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(values, f1_scores, marker="o")
    axes[0].set_title("F1 vs " + param_name)
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel("F1")

    axes[1].plot(values, state_counts, marker="o", color="orange")
    axes[1].set_title("State Count")
    axes[1].set_xlabel(param_name)

    axes[2].plot(values, densities, marker="o", color="green")
    axes[2].set_title("Transition Density")
    axes[2].set_xlabel(param_name)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
