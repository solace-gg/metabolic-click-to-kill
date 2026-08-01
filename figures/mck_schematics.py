#!/usr/bin/env python3
"""Reproducible schematic figures for the MCK thesis (matplotlib vector -> PNG).

Generates the two computed schematics used in the thesis:
  * fig_third_gate_mechanism.png - the third gate as a location gate
  * fig_docking_pose.png         - the glucuronide-masked sugar in beta-glucuronidase
Conceptual illustrations (reach geometry, MCK-E triple-bind, etc.) are produced
separately as vector art and are not generated here.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
import numpy as np, os

OUT = os.path.dirname(os.path.abspath(__file__))   # write beside this script
NK = "#2c6fbb"; TUM = "#b5651d"; COAT = "#d9c6a5"; WELD = "#c0392b"; INK = "#222"; GREY = "#666"


# ---------------------------------------------------------------- THIRD-GATE (LOCATION) MECHANISM
def third_gate():
    fig, ax = plt.subplots(figsize=(12.5, 4.6)); ax.set_xlim(0, 12.4); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(6.2, 4.7, "The third gate as a location gate: extracellular uncaging then local uptake",
            ha="center", fontsize=11, fontweight="bold")
    boxes = [("Masked 3G-AAM\n(charged, membrane-\nimpermeant)", "#fbe9e7", "#c0392b"),
             ("Reaches tumour\ninterstitium via leaky\nvasculature (EPR)", "#eef3f8", "#2c6fbb"),
             ("Extracellular trigger\n(NIR / focused ultrasound,\nor beta-glucuronidase)\ncleaves the mask", "#e8f0e8", "#2e7d32"),
             ("Self-immolation:\n1,6-elimination + CO2\n-> C6-unmasked sugar", "#fff6e5", "#b5651d"),
             ("Local uptake; HDAC x\ncathepsin-L display\nthe azide", "#eef3f8", "#1b4a7a")]
    x = 0.3; w = 1.95; y = 2.2
    for i, (t, fc, ec) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, y), w, 1.5, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.5))
        ax.text(x + w / 2, y + 0.75, t, ha="center", va="center", fontsize=8, color="#222")
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + 0.32, y + 0.75), xytext=(x + w + 0.02, y + 0.75),
                        arrowprops=dict(arrowstyle="-|>", color="#444", lw=2))
        x += w + 0.34
    ax.text(6.2, 1.4, "Distant normal tissue: no extracellular trigger -> no release -> not labelled (selectivity is against distant tissue).",
            ha="center", fontsize=8.5, style="italic", color=GREY)
    ax.text(6.2, 0.9, "Freed sugar labels a ~40-350 um bystander penumbra, still filtered by the HDAC x cathepsin-L gate.",
            ha="center", fontsize=8.5, style="italic", color=GREY)
    fig.savefig(os.path.join(OUT, "fig_third_gate_mechanism.png"), dpi=160, facecolor="w")
    plt.close(fig); print("fig_third_gate_mechanism.png")


# ---------------------------------------------------------------- DOCKING POSE (schematic; endogenous route only)
def docking():
    fig, ax = plt.subplots(figsize=(7, 5)); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(5, 9.5, "Glucuronide-masked sugar in the beta-glucuronidase active site",
            ha="center", fontsize=10.5, fontweight="bold")
    pocket = Polygon([(2, 2), (8, 2), (7.2, 6.5), (2.8, 6.5)], closed=True, fc="#eef3f8", ec="#2c6fbb", lw=1.5, alpha=0.7)
    ax.add_patch(pocket); ax.text(5, 2.4, "beta-glucuronidase pocket (PDB 3HN3)", ha="center", fontsize=8, color="#2c6fbb")
    ax.add_patch(Circle((4.2, 3.4), 0.18, fc="#2e7d32")); ax.add_patch(Circle((5.2, 3.2), 0.18, fc="#2e7d32"))
    ax.text(4.7, 2.9, "catalytic Glu451/Glu540", ha="center", fontsize=8, color="#2e7d32")
    ax.add_patch(Circle((4.7, 4.0), 0.4, fc="#e8f0e8", ec="#2e7d32", lw=1.5)); ax.text(4.7, 4.0, "GlcA", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(4.7, 3.6), xytext=(4.7, 3.3), arrowprops=dict(arrowstyle="-", color=WELD, lw=1.5))
    ax.text(5.5, 3.75, "2.65 A", ha="left", fontsize=8.5, color=WELD, fontweight="bold")
    ax.plot([4.9, 6.5], [4.4, 7.8], color=TUM, lw=2.5, solid_capstyle="round")
    ax.add_patch(Circle((6.7, 8.0), 0.5, fc="#fff6e5", ec=TUM, lw=1.5)); ax.text(6.7, 8.0, "Ac4\nsugar", ha="center", va="center", fontsize=7.5)
    ax.text(7.4, 8.0, "bulky acetylated tail\nprojects out of the pocket\n(no steric exclusion)", ha="left", va="center", fontsize=8)
    ax.text(5, 0.9, "Docking (Vina, exhaustiveness 16, 3 seeds): the glucuronide contacts the catalytic pair (2.65 A),\nconfirming cleavage-competence; the affinity is not read as tighter binding than the substrate.\nApplies to the endogenous glucuronide route only.",
            ha="center", fontsize=8, style="italic", color=GREY)
    fig.savefig(os.path.join(OUT, "fig_docking_pose.png"), dpi=160, facecolor="w"); plt.close(fig); print("fig_docking_pose.png")


if __name__ == "__main__":
    third_gate(); docking()
