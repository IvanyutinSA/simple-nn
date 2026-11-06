use simple_nn::{
    matrix::Matrix,
    parser::simple_csv::parse_csv,
    classic::{
        linreg
    }
};


fn main() {
    let a = Matrix::fill(3, 2, 1.);
    let b = Matrix::fill(3, 2, 2.);
    let c = Matrix::fill(2, 6, 5.);
    println!("a+b");
    println!("{:?}", &a+&b);
    println!("a-b");
    println!("{:?}", &a-&b);
    println!("a*c");
    println!("{:?}", &a*&c);
    println!("a.*c");
    println!("{:?}", a.mul_pointwise(&b).unwrap());
    let parsed = parse_csv("datasets/Salary_dataset.csv", ',').unwrap();
    let data = Matrix::from(parsed);
    let x = data.project_cols(&vec![1]);
    let y = data.project_cols(&vec![2]);

    let mut weights = linreg::LinRegWeights::default();
    linreg::fit(&mut weights, &x, &y, 50000000);
}
