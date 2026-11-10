use crate::matrix::Matrix;


pub struct NNSequence {
    pub seq: Vec<NNComponent>,
}


pub enum NNComponent {
    ActivationFunction(Box::<dyn Activation>),
    Layer(Box::<dyn Layer>),
}


pub trait Activation {
    fn eval(&self, x: &Matrix) -> Matrix;
    fn deriv(&self, x: &Matrix) -> Matrix;
}


pub trait Layer {
    fn weights(&self) -> Vec<Matrix>;
    fn update(&mut self, grad: GeneralGrad);
    fn forward(&self, x: &Matrix) -> Matrix;
    fn backward(&self, x: &Matrix, grad: &Matrix)
        -> (Matrix, GeneralGrad);
}

pub trait Loss {
    fn eval(&self, y: &Matrix, yhat: &Matrix) -> f64;
    fn deriv(&self, y: &Matrix, yhat: &Matrix) -> Matrix;
}

#[derive(Debug)]
pub struct LinearLayer {
    // nxk
    pub w: Matrix,
    // 1xk
    pub w0: Matrix,
}

impl LinearLayer {
    pub fn new(n_in: usize, k_out: usize) -> Self {
        // LinearLayer { 
        //     w: Matrix::std(n_in, k_out),
        //     w0: Matrix::std(1, k_out),
        // }
        LinearLayer { 
            // w: Matrix::fill(n_in, k_out, (2./n_in as f64).sqrt()),
            // w0: Matrix::fill(1, k_out, 0.),
            w: Matrix::he_init(n_in, k_out),
            w0: Matrix::he_init(1, k_out),
        }
    }
}

#[derive(Debug)]
pub struct GeneralGrad {
    pub grads: Vec<Matrix>,
}

impl Layer for LinearLayer {
    fn weights(&self) -> Vec<Matrix> {
        vec![self.w.clone()]
    }

    // x: mxn
    // x.T@w: mxk
    // out: mxk
    fn forward(&self, x: &Matrix) -> Matrix {
        &(x*&self.w)+&self.w0.broadcast_rows(x.m)
    }

    // x: mxn
    // grad: mxk
    // x.T@grad.T: nxk
    // grad_out: mxn
    // out: (GradOut, GradLayer) 
    fn backward(&self, x: &Matrix, grad: &Matrix)
        -> (Matrix, GeneralGrad) {
        let grad_w = &x.transpose()*grad;
        let grad_w0 = grad.sum_rows();
        let grad_layer = GeneralGrad {
            grads: vec![grad_w, grad_w0]
        };
        let grad_out = grad * &self.w.transpose();
        (grad_out, grad_layer)
    }

    fn update(&mut self, mut grad: GeneralGrad) {
        self.w0 = &self.w0-&grad.grads.pop().unwrap();
        self.w = &self.w-&grad.grads.pop().unwrap();
    }
}

pub struct ReLU;

impl Activation for ReLU {
    fn eval(&self, x: &Matrix) -> Matrix {
        x.apply(|x| {
            if x > 0. {
                x
            } else {
                0.
            }
        })
    }
    fn deriv(&self, x: &Matrix) -> Matrix {
        x.apply(|x| {
            if x > 0. {
                1.
            } else {
                0.
            }
        })
    }
}

pub struct MSE;

impl Loss for MSE {
    fn eval(&self, y: &Matrix, yhat: &Matrix) -> f64 {
        let m = y.m as f64;
        (y-yhat).apply(|v| v.powi(2)/m).inner.iter().sum()
    }

    fn deriv(&self, y: &Matrix, yhat: &Matrix) -> Matrix {
        let m = y.m as f64;
        (y-yhat).apply(|v| v*-2./m)
    }
}

pub fn fit(x: &Matrix, y: &Matrix) -> NNSequence {
    let mut _mlp = NNSequence {
        seq: vec![
            NNComponent::Layer(Box::new(LinearLayer::new(3, 3))),
            NNComponent::ActivationFunction(Box::new(ReLU)),
            NNComponent::Layer(Box::new(LinearLayer::new(3, 1))),
        ],
    };
    let mut mlp = NNSequence {
        seq: vec![
            NNComponent::Layer(Box::new(LinearLayer::new(3, 8))),
            NNComponent::ActivationFunction(Box::new(ReLU)),
            NNComponent::Layer(Box::new(LinearLayer::new(8, 16))),
            NNComponent::ActivationFunction(Box::new(ReLU)),
            NNComponent::Layer(Box::new(LinearLayer::new(16, 32))),
            NNComponent::ActivationFunction(Box::new(ReLU)),
            NNComponent::Layer(Box::new(LinearLayer::new(32, 16))),
            NNComponent::ActivationFunction(Box::new(ReLU)),
            NNComponent::Layer(Box::new(LinearLayer::new(16, 8))),
            NNComponent::ActivationFunction(Box::new(ReLU)),
            NNComponent::Layer(Box::new(LinearLayer::new(8, 4))),
            NNComponent::ActivationFunction(Box::new(ReLU)),
            NNComponent::Layer(Box::new(LinearLayer::new(4, 1))),
        ]
    };
    let lr = 0.001;
    let loss_f = MSE;
    let epochs = 50000;
    let mut last_loss = -1.;
    let l2_lambda = 1./x.m as f64;

    // l2 is l*sum(wegihts^2)

    for epoch in 0..epochs {
        let mut us = vec![x.clone()];
        for component in mlp.seq.iter() {
            match component {
                NNComponent::Layer(layer) => {
                    us.push(layer.forward(us.last().unwrap()));
                }
                NNComponent::ActivationFunction(func) => {
                    us.push(func.eval(us.last().unwrap()));
                }
            }
        }
        let loss = loss_f.eval(y, &us.last().unwrap());
        
        // l2 calcs

        let l2_loss = l2_lambda*mlp.seq.iter().filter_map(|component| {
            match component {
                NNComponent::Layer(layer) => {
                    Some(layer.weights())
                },
                _ => None,
            }
        }).flatten().map(|mat: Matrix| {
            mat.inner.iter().map(|x: &f64| x.powi(2)).collect::<Vec<f64>>()
        }).flatten().sum::<f64>();

        let loss = loss + l2_loss;
        println!("[{}], loss: {}",
            epoch, loss);
        if (last_loss - loss).abs() < 0.00000001 {
            break;
        }
        last_loss = loss;

        let mut grad = loss_f.deriv(y, &us.pop().unwrap());
        for mut component in mlp.seq.iter_mut().rev() {
            match component {
                NNComponent::Layer(layer) => {
                    let (new_grad, mut gen_grad) = 
                        layer.backward(&us.pop().unwrap(), &grad);
                    grad = new_grad;
                    gen_grad.grads = gen_grad.grads.into_iter()
                        .map(|mat| {
                            mat.apply(|x| x*lr)
                        }).collect();
                    layer.update(gen_grad);
                    let l2grad = GeneralGrad {
                        grads: vec![
                            // -2.*w
                            layer.weights()[0].apply(|x| 2.*x*l2_lambda*lr),
                            Matrix::fill(1, layer.weights()[0].n, 0.)
                        ]
                    };
                    layer.update(l2grad);
                }
                NNComponent::ActivationFunction(func) => {
                    grad = func.deriv(&us.pop().unwrap())
                        .mul_pointwise(&grad).unwrap();
                }
            }
        }
    }
    mlp
}
