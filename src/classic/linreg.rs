use crate::matrix::Matrix;

#[derive(Default)]
pub struct LinRegWeights {
    // mx1
    pub a: Matrix,
    pub b: f64,
}

// data:   mxn
// target: mx1
pub fn fit(
    weights: &mut LinRegWeights,
    data: &Matrix,
    target: &Matrix,
    epoches: usize
    ) {
    let lr = 0.0000001;
    let a = Matrix::std(data.n, 1);
    let b: f64 = Matrix::std(1, 1).get(0, 0).unwrap();
    weights.a = a;
    weights.b = b;

    for i in 0..epoches {
        // mx1
        let yhat = data*&weights.a+b;

        // loss is MSE
        // MSE(y, y_hat) = sum((y-yhat)**2)
        let loss: f64 = (target-&yhat)
            .apply(|x| x.powi(2))
            .inner
            .iter()
            .sum();
        let loss = loss/data.m as f64;

        println!("[{}] loss: {}", i, loss);

        // vector of -2*(y[i]-yhat[i])*(dyhat/dtheta)[i]
        // each row respects to ith sum component
        let dl_dyhat = -2.*&(target-&yhat);

        let dyhat_db = Matrix::fill(yhat.m, 1, 1.);

        let dl_db: f64 = dl_dyhat.mul_pointwise(&dyhat_db).unwrap()
            .inner
            .iter()
            .sum();

        let dl_da = &data.transpose()*&dl_dyhat;

        weights.a = &weights.a - &(lr*&dl_da);
        weights.b = weights.b - lr*dl_db;
    }
}
