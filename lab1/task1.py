import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, expon


def generate_samples(shape, scale, n_samples, sample_size):
    """Генерирует выборки из гамма-распределения."""
    return gamma.rvs(shape, scale=scale, size=(n_samples, sample_size))


def compute_statistics(samples, shape, scale):
    """Вычисляет статистики для выборок."""
    means = np.mean(samples, axis=1)  # Выборочное среднее
    variances = np.var(
        samples, axis=1, ddof=1
    )  # Выборочная дисперсия с поправкой Бесселя (несмещённая)
    quantiles_05 = np.quantile(samples, 0.5, axis=1)  # Выборочная квантиль порядка 0.5
    nF_X2 = len(samples[0]) * gamma.cdf(
        np.sort(samples, axis=1)[:, 1], shape, scale=scale
    )
    n1_F_Xn = len(samples[0]) * (
        1 - gamma.cdf(np.sort(samples, axis=1)[:, -1], shape, scale=scale)
    )
    return means, variances, quantiles_05, nF_X2, n1_F_Xn


def plot_histograms(
    means, variances, quantiles_05, nF_X2, n1_F_Xn, shape, scale, sample_size
):
    """Строит гистограммы распределений статистик."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    bins = 100

    # Гистограмма и теоретическая плотность для выборочного среднего
    x_means = np.linspace(min(means), max(means), 100)
    pdf_means = norm.pdf(
        x_means, shape * scale, np.sqrt(shape * scale**2 / sample_size)
    )
    axs[0, 0].hist(
        means, bins=bins, density=True, alpha=0.5, color="green", label="Empirical"
    )
    axs[0, 0].plot(x_means, pdf_means, "r--", lw=2, label="Theoretical")
    axs[0, 0].set_title("Sample Mean")
    axs[0, 0].legend()

    # Гистограмма и теоретическая плотность для выборочной дисперсии
    theoretical_mean = shape * (scale**2)
    theoretical_variance = (2 * (shape * (scale**2)) ** 2) / (sample_size - 1)

    x_vars = np.linspace(np.min(variances), np.max(variances), 100)
    pdf_vars = norm.pdf(
        x_vars, loc=theoretical_mean, scale=np.sqrt(theoretical_variance)
    )
    axs[0, 1].hist(
        variances, bins=bins, density=True, alpha=0.5, color="blue", label="Empirical"
    )
    axs[0, 1].plot(x_vars, pdf_vars, "m--", lw=2, label="Theoretical")
    axs[0, 1].set_title("Sample Variance")
    axs[0, 1].legend()

    # Гистограмма для выборочной квантили порядка 0.5 (медианы)
    # Для медианы используем ядерную оценку плотности# Рассчитываем среднее значение и стандартное отклонение выборочных медиан
    mean_median = np.mean(quantiles_05)
    std_median = np.std(quantiles_05, ddof=1)
    x_median = np.linspace(min(quantiles_05), max(quantiles_05), 100)
    pdf_median = norm.pdf(x_median, mean_median, std_median)
    axs[0, 2].hist(
        quantiles_05,
        bins=bins,
        density=True,
        alpha=0.5,
        color="yellow",
        label="Empirical",
    )
    axs[0, 2].plot(
        x_median,
        pdf_median,
        "r--",
        lw=2,
        label="Theoretical",
    )
    axs[0, 2].set_title("Sample Quantile (0.5)")
    axs[0, 2].legend()

    # -> Г(2, 1)
    axs[1, 0].hist(
        nF_X2, bins=bins, density=True, alpha=0.5, color="cyan", label="Empirical"
    )
    axs[1, 0].plot(
        np.linspace(0, 10, 100),
        gamma.pdf(np.linspace(0, 10, 100), 2, scale=1),
        color="black",
        label="Theoretical",
    )
    axs[1, 0].set_title("nF(X_(2))")
    axs[1, 0].legend()

    # -> Г(1, 1) = Exp(1)
    axs[1, 1].hist(
        n1_F_Xn, bins=bins, density=True, alpha=0.5, color="cyan", label="Empirical"
    )
    axs[1, 1].plot(
        np.linspace(0, 10, 100),
        expon.pdf(np.linspace(0, 10, 100)),
        color="black",
        label="Theoretical",
    )
    axs[1, 1].set_title("n(1-F(X_(n)))")
    axs[1, 1].legend()

    # Остальные графики уже не нужны
    axs[1, 2].set_visible(False)

    plt.tight_layout()
    plt.show()


def print_statistics(means, variances, quantiles_05, nF_X2, n1_F_Xn):
    """Выводит числовые характеристики статистик."""
    print(
        f"Sample Mean: Mean={np.mean(means):.2f}, Std={np.std(means):.2f}, Median={np.median(means):.2f}"
    )
    print(
        f"Sample Variance: Mean={np.mean(variances):.2f}, Std={np.std(variances):.2f}, Median={np.median(variances):.2f}"
    )
    print(
        f"Sample Quantile (0.5) (e.g. Median): Mean={np.mean(quantiles_05):.2f}, Std={np.std(quantiles_05):.2f}, Median={np.median(quantiles_05):.2f}"
    )
    print(
        f"nF(X_(2)): Mean={np.mean(nF_X2):.2f}, Std={np.std(nF_X2):.2f}, Median={np.median(nF_X2):.2f}"
    )
    print(
        f"n(1-F(X_(n))): Mean={np.mean(n1_F_Xn):.2f}, Std={np.std(n1_F_Xn):.2f}, Median={np.median(n1_F_Xn):.2f}"
    )


def main():
    # Параметры гамма-распределения
    shape, scale = 14.0, 1.0

    # Генерация выборок
    n_samples, sample_size = 1000, 10000
    samples = generate_samples(shape, scale, n_samples, sample_size)

    # Вычисление статистик
    means, variances, quantiles_05, nF_X2, n1_F_Xn = compute_statistics(
        samples, shape, scale
    )

    # Построение графиков
    plot_histograms(
        means, variances, quantiles_05, nF_X2, n1_F_Xn, shape, scale, sample_size
    )

    # Вывод статистик
    print_statistics(means, variances, quantiles_05, nF_X2, n1_F_Xn)


if __name__ == "__main__":
    main()
