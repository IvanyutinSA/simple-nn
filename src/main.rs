use simple_nn::{
    matrix::Matrix,
    parser::simple_csv::parse_csv,
    classic::{
        linreg
    },
    statistics as st,
    metrics,
};


fn main() {
    let parsed = parse_csv("datasets/Salary_dataset.csv", ',').unwrap();
    let data = Matrix::from(parsed);
    let data = data.project_cols(&vec![1,2]);

    let x = data.project_cols(&vec![0]);
    let y = data.project_cols(&vec![1]);

    let sigma2 = vec![
        st::variance(&x.project_cols(&vec![0]).inner),
        st::variance(&y.project_cols(&vec![0]).inner),
    ];

    let mu = vec![
        st::mean(&x.project_cols(&vec![0]).inner),
        st::mean(&y.project_cols(&vec![0]).inner),
    ];

    println!("{:?}", sigma2);
    println!("{:?}", mu);

    let x = x.apply(|x| (x-mu[0])/sigma2[0].sqrt());
    let y = y.apply(|y| (y-mu[1])/sigma2[1].sqrt());

    // let test_rows = vec![4, 15, 16, 22, 27];
    let test_rows = (0..30).collect();

    let mut weights = linreg::LinRegWeights::default();
    linreg::fit(&mut weights, &x, &y, 1000);
    println!("{:?}", weights);

    println!("{:?}", metrics::r2(
            &y.project_rows(&test_rows).inner,
            &linreg::eval(&weights, &x.project_rows(&test_rows)).inner,
            ));
}
