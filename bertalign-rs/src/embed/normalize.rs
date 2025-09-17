use candle_core::Tensor;
use candle_nn::Module;

pub struct Normalize;

impl Module for Normalize {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // Divide each value by the L2 Norm - square root fot he sum of the squares of each component
        Ok(xs.broadcast_div(&xs.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }
}
