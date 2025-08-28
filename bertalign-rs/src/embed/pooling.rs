pub trait Pooling {
    fn pool();
}

struct MeanPooling;

impl Pooling for MeanPooling {
    fn pool() {}
}
