use crate::statistics;

pub fn r2(y: &[f64], yhat: &[f64]) -> f64 {
    let rsm: f64 = y.iter().zip(yhat).map(|(x, y)| (x-y).powi(2)).sum();
    let y_mean = statistics::mean(y);
    let tss: f64 = y.iter().map(|x| (x-y_mean).powi(2)).sum();
    1.-rsm/tss
}
