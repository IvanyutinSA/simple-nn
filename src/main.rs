use simple_nn::{
    matrix::Matrix,
    parser::simple_csv::parse_csv,
    classic::{
        linreg,
        mlp:: {
            self,
            NNComponent,
        },
    },
    statistics as st,
    metrics,
};

fn main() {
    let parsed = parse_csv("datasets/advertising.csv", ',').unwrap();
    let data = Matrix::from(parsed);
    // let data = data.project_cols(&vec![1,2]);

    let mut x = data.project_cols(&vec![0, 1, 2]);
    let y = data.project_cols(&vec![3]);

    let sigma2 = vec![
        st::variance(&x.project_cols(&vec![0]).inner),
        st::variance(&x.project_cols(&vec![1]).inner),
        st::variance(&x.project_cols(&vec![2]).inner),
        st::variance(&y.project_cols(&vec![0]).inner),
    ];

    let mu = vec![
        st::mean(&x.project_cols(&vec![0]).inner),
        st::mean(&x.project_cols(&vec![1]).inner),
        st::mean(&x.project_cols(&vec![2]).inner),
        st::mean(&y.project_cols(&vec![0]).inner),
    ];

    println!("{:?}", sigma2);
    println!("{:?}", mu);

    // let x = x.apply(|x| (x-mu[0])/sigma2[0].sqrt());
    for i in 0..x.m {
        for j in 0..x.n {
            x.set(i, j, (x.get(i,j).unwrap()-mu[j])/sigma2[j].sqrt()).unwrap();
        }
    }


    let y = y.apply(|y| (y-mu[1])/sigma2[1].sqrt());

    // let test_rows = vec![4, 15, 16, 22, 27];
    let test_rows = (0..30).collect();

    let nn = mlp::fit(&x, &y);
    let mut us = vec![x.project_rows(&test_rows).clone()];
    for component in nn.seq.iter() {
        match component {
            NNComponent::Layer(layer) => {
                us.push(layer.forward(us.last().unwrap()));
            }
            NNComponent::ActivationFunction(func) => {
                us.push(func.eval(us.last().unwrap()));
            }
        }
    }

    println!("{:?}", metrics::r2(
            &y.project_rows(&test_rows).inner,
            &us.last().unwrap().inner,
            ));
}

fn main_linreg() {
    let parsed = parse_csv("datasets/advertising.csv", ',').unwrap();
    let data = Matrix::from(parsed);
    // let data = data.project_cols(&vec![1,2]);

    let mut x = data.project_cols(&vec![0, 1, 2]);
    let y = data.project_cols(&vec![3]);

    let sigma2 = vec![
        st::variance(&x.project_cols(&vec![0]).inner),
        st::variance(&x.project_cols(&vec![1]).inner),
        st::variance(&x.project_cols(&vec![2]).inner),
        st::variance(&y.project_cols(&vec![0]).inner),
    ];

    let mu = vec![
        st::mean(&x.project_cols(&vec![0]).inner),
        st::mean(&x.project_cols(&vec![1]).inner),
        st::mean(&x.project_cols(&vec![2]).inner),
        st::mean(&y.project_cols(&vec![0]).inner),
    ];

    println!("{:?}", sigma2);
    println!("{:?}", mu);

    // let x = x.apply(|x| (x-mu[0])/sigma2[0].sqrt());
    for i in 0..x.m {
        for j in 0..x.n {
            x.set(i, j, (x.get(i,j).unwrap()-mu[j])/sigma2[j].sqrt()).unwrap();
        }
    }


    let y = y.apply(|y| (y-mu[1])/sigma2[1].sqrt());

    // let test_rows = vec![4, 15, 16, 22, 27];
    let test_rows = (0..30).collect();

    let mut weights = linreg::LinRegWeights::default();
    linreg::fit(&mut weights, &x, &y, 1000000);
    println!("{:?}", weights);

    println!("{:?}", metrics::r2(
            &y.project_rows(&test_rows).inner,
            &linreg::eval(&weights, &x.project_rows(&test_rows)).inner,
            ));
}
