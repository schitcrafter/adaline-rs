use rand::{
    distributions::{Distribution, Uniform},
    thread_rng,
};

use crate::Classification;

#[derive(Debug)]
pub struct Adaline {
    weights: [f32; 3],
    /// The constant "mu", which will determine
    /// how much the weights will be changed
    /// while training
    train_constant: f32,
}

impl Adaline {
    pub fn new(weights: [f32; 3], train_constant: f32) -> Adaline {
        Adaline {
            weights,
            train_constant,
        }
    }

    pub fn new_random(train_constant: f32) -> Adaline {
        let distribution = Uniform::new_inclusive(-1.0, 1.0);
        let mut gen = thread_rng();

        let weights: Vec<_> = (0..3).map(|_| distribution.sample(&mut gen)).collect();

        Adaline {
            weights: weights.try_into().unwrap(),
            train_constant,
        }
    }

    pub fn classify_multiple_percent(&self, class: &[Classification]) -> f32 {
        self.classify_multiple(class) as f32 / class.len() as f32
    }

    /// Returns how many of the classifications were classified
    /// correctly
    pub fn classify_multiple(&self, classifications: &[Classification]) -> u32 {
        let correct_classifications = classifications
            .iter()
            .filter(|classific| self.classify(&classific.x) == classific.classification)
            .count();

        correct_classifications as u32
    }

    /// Get the dot product of the weights and [1, ..x_vec]
    /// This is the classification without the delta function
    /// (not just -1 or 1, but a real number)
    pub fn classify_no_alpha(&self, x_vec: &[f32; 2]) -> f32 {
        let mut sum = self.weights[0];

        for (i, x) in x_vec.iter().enumerate() {
            sum += x * self.weights[i + 1]
        }

        sum
    }

    /// The x vector here shouldn't start with 1,
    /// the 1 at the beginning is emulated.
    pub fn classify(&self, x_vec: &[f32; 2]) -> bool {
        let sum = self.classify_no_alpha(x_vec);
        sum.is_sign_positive()
    }

    /// Use the classification to improve our own weights
    pub fn train(&mut self, actual_class: &Classification) {
        let guessed_class = self.classify_no_alpha(&actual_class.x);
        let y = if actual_class.classification {
            1.0
        } else {
            -1.0
        };
        let delta = y - guessed_class;

        // println!("Guess {guessed_class} is off by delta={delta}");

        let mut weight_deltas = [0f32; 3];

        // weights[0] corresponds to x[0], which should be 1 but is ommitted here
        weight_deltas[0] = delta;

        for (i, x) in actual_class.x.iter().enumerate() {
            weight_deltas[i + 1] = delta * x;
        }

        // Add weight deltas to current weight's, weighed with the train_constant
        let new_weights: Vec<_> = self
            .weights
            .iter()
            .enumerate()
            .map(|(i, weight)| weight + self.train_constant * weight_deltas[i])
            .collect();

        self.weights = new_weights.try_into().unwrap();
    }

    pub fn weights(&self) -> &[f32; 3] {
        &self.weights
    }
}
