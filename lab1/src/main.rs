use nalgebra::DMatrix;
use ndarray::{Array1, Array2, Axis};
use ndarray_stats::QuantileExt;
use plotters::prelude::*;
use rand::distributions::{Gamma, Distribution};
use statrs::distribution::{Continuous, Exponential, Normal, StudentsT};

fn main() {
    // Параметры гамма-распределения
    let shape = 2.0;
    let scale = 1.0;

    // Генерация выборок
    let n_samples = 1000;
    let sample_size = 100;
    let mut samples = DMatrix::from_fn(n_samples, sample_size, |_, _| {
        let gamma = Gamma::new(shape, scale).unwrap();
        gamma.sample(&mut rand::thread_rng())
    });

    // Вычисление статистик
    let means = samples.column_mean();
    let variances = samples.column_variance();
    let medians = samples
        .as_slice()
        .chunks(sample_size)
        .map(|row| row.quantile_mut(0.5))
        .collect::<Vec<_>>();
    let sorted_samples = samples.as_mut_slice().sort_unstable_by(|a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut nf_x2 = Vec::new();
    let mut n1_f_xn = Vec::new();
    for i in 0..n_samples {
        let gamma = Gamma::new(shape, scale).unwrap();
        nf_x2.push(sample_size as f64 * gamma.cdf(sorted_samples[i * sample_size + 1]));
        n1_f_xn.push(sample_size as f64 * (1.0 - gamma.cdf(sorted_samples[i * sample_size + sample_size - 1])));
    }

    // Построение графиков
    let root_area = BitMapBackend::new("output.png", (1200, 800)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root_area)
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..3.0, 0.0..1.0)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart.draw_series(
        Histogram::vertical(&chart)
            .style(BLUE.filled())
            .margin(0)
            .data(means.as_slice().iter().map(|&x| (x, 0.1))),
    ).unwrap();

    chart.draw_series(
        LineSeries::new(
            (0..100).map(|i| i as f64 * 0.03),
            (0..100).map(|i| {
                let x = i as f64 * 0.03;
                Normal::new(shape * scale, (shape * scale.powi(2) / sample_size as f64).sqrt())
                    .unwrap()
                    .pdf(x)
            }),
            &RED,
        )
    ).unwrap();

    // Аналогично построить остальные графики...

    // Вывод статистик
    println!("Sample Mean: Mean={:.2}, Std={:.2}, Median={:.2}", 
             means.mean(), means.std_dev(), medians[n_samples / 2]);

    // Аналогично вывести остальные статистики...
}
