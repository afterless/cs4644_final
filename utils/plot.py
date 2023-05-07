import matplotlib.pyplot as plt


def plot_interp_acc(
    lambdas,
    train_loss_interp_naive,
    test_loss_interp_naive,
    train_loss_interp_clever,
    test_loss_interp_clever,
):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        lambdas,
        train_loss_interp_naive,
        linestyle="dashed",
        color="tab:blue",
        alpha=0.5,
        linewidth=2,
        label="Train, naive interp.",
    )
    ax.plot(
        lambdas,
        test_loss_interp_naive,
        linestyle="dashed",
        color="tab:orange",
        alpha=0.5,
        linewidth=2,
        label="Test, naive interp.",
    )
    ax.plot(
        lambdas,
        train_loss_interp_clever,
        linestyle="solid",
        color="tab:blue",
        linewidth=2,
        label="Train, permuted interp.",
    )
    ax.plot(
        lambdas,
        test_loss_interp_clever,
        linestyle="solid",
        color="tab:orange",
        linewidth=2,
        label="Test, permuted interp.",
    )
    ax.set_xlabel("$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel("Loss")
    ax.set_title("Interpolated Loss")
    ax.legend(loc="lower right", framealpha=0.5)
    fig.tight_layout()
    return fig
