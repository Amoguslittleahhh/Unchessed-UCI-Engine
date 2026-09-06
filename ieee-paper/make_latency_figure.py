import matplotlib.pyplot as plt

moves = list(range(8, 18))
elo = [2411, 2468, 2469, 2531, 2580, 2580, 2616, 2623, 2650, 2634]
confidence = [381, 364, 318, 298, 278, 248, 233, 211, 199, 183]
labels = [
    "steady", "trending up", "erratic", "erratic", "erratic",
    "erratic", "erratic", "erratic", "erratic", "ceiling"
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
})
fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=220)
ax.errorbar(moves, elo, yerr=confidence, fmt="o-", color="#154c79",
            ecolor="#7aa6c2", elinewidth=1.1, capsize=3, linewidth=1.8,
            markersize=4.5, label="Estimated opponent Elo ± confidence")
ax.axvline(17.5, color="#b22222", linestyle="--", linewidth=1.2,
           label="FULL transition after move 17")
ax.axhline(2500, color="#777777", linestyle=":", linewidth=1.0,
           label="2500-Elo reference")
ax.set_xlim(7.5, 18.5)
ax.set_ylim(1900, 3100)
ax.set_xticks(range(8, 19))
ax.set_xlabel("Opponent move number")
ax.set_ylabel("Estimated Elo (rating points)")
ax.set_title("Rating estimate rose steadily while classification remained erratic")
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="upper left", frameon=True, fontsize=7.8)
fig.tight_layout()
fig.savefig("/home/ubuntu/unchessed-repo/ieee-paper/detection_latency.pdf", bbox_inches="tight")
fig.savefig("/home/ubuntu/unchessed-repo/ieee-paper/detection_latency.png", bbox_inches="tight")
