use simple_nn::{
    matrix::Matrix,
    parser::simple_csv::parse_csv,
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
    println!("{:?}", Matrix::from(parsed));
}
