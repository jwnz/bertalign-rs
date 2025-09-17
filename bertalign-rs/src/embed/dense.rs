use candle_core::Tensor;
use candle_nn::{linear_b, Linear, Module, VarBuilder};
use serde::Deserialize;

use crate::error::DenseError;

#[derive(Deserialize)]
pub enum Activation {
    #[serde(alias = "Tanh")]
    #[serde(alias = "torch.nn.modules.activation.Tanh")]
    Tanh,
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = match self {
            Activation::Tanh => xs.tanh()?,
        };

        Ok(xs)
    }
}

pub struct Dense {
    linear: Linear,
    activation_function: Activation,
}

impl Dense {
    pub fn from_config(
        vb: VarBuilder,
        config: SentenceTransformersDenseConfig,
    ) -> Result<Dense, DenseError> {
        Ok(Self {
            linear: linear_b(
                config.in_features,
                config.out_features,
                config.bias,
                vb.pp("linear"),
            )?,
            activation_function: config.activation_function,
        })
    }
}

impl Module for Dense {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        Ok(self
            .activation_function
            .forward(&self.linear.forward(&xs)?)?)
    }
}

#[derive(Deserialize)]
pub struct SentenceTransformersDenseConfig {
    in_features: usize,
    out_features: usize,
    bias: bool,
    activation_function: Activation,
}
