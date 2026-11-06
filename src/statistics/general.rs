// E[x^k]
pub fn raw_moment(order: i32, data: &[f64]) -> f64 {
    assert!(order > 0);
    let bullshit: f64 = data.iter().map(|x| x.powi(order)).sum();
    bullshit/(data.len() as f64)
}

// E[(x-mu)^k]
pub fn central_moment(order: i32, data: &[f64]) -> f64 {
    assert!(order > 0);
    let mu = raw_moment(1, data);
    let bullshit: f64 = data.iter().map(|x| (x-mu).powi(order)).sum();
    bullshit/(data.len() as f64)
}

// E[x]
pub fn mean(data: &[f64]) -> f64 {
    raw_moment(1, data)
}

// E[(x-mu)^2]
pub fn variance(data: &[f64]) -> f64 {
    central_moment(2, data)
}
