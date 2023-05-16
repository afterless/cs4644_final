import matplotlib.pyplot as plt


def plot_interp_acc(
    lam,
    train_loss_interp_naive,
    test_loss_interp_naive,
    train_loss_interp_clever,
    test_loss_interp_clever,
):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        lam,
        train_loss_interp_naive,
        linestyle="dashed",
        color="tab:blue",
        alpha=0.5,
        linewidth=2,
        label="Train Naive Loss",
    )
    ax.plot(
        lam,
        test_loss_interp_naive,
        linestyle="dashed",
        color="tab:orange",
        alpha=0.5,
        linewidth=2,
        label="Test Naive Loss",
    )
    ax.plot(
        lam,
        train_loss_interp_clever,
        linestyle="soild",
        color="tab:blue",
        linewidth=2,
        label="Train Permute Loss",
    )
    ax.plot(
        lam,
        test_loss_interp_clever,
        linestyle="soild",
        color="tab:orange",
        linewidth=2,
        label="Test Permute Loss",
    )

    ax.set_xlabel("$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel("Loss")

    ax.set_title(f"Loss Interpolation between two models")
    ax.legend(loc="upper right", framealpha=0.5)
    fig.tight_layout()
    return fig
