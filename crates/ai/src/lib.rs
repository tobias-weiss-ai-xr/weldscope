#[cfg(test)]
mod tests {
    use super::*;

    const W: &str = r#"{
        "classes": ["ok", "incomplete", "pore"],
        "weights": [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ],
        "bias": [0.0, 0.0, 0.0]
    }"#;

    #[test]
    fn loads_and_predicts_argmax() {
        let c = Classifier::load(W).unwrap();
        let (cls, _) = c.predict(&[0.0; 8]);           // all scores 0 => "ok"
        assert_eq!(cls, "ok");
        let (cls, _) = c.predict(&[0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]); // incomplete
        assert_eq!(cls, "incomplete");
        let (cls, _) = c.predict(&[0.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0]); // pore
        assert_eq!(cls, "pore");
    }

    #[test]
    fn rejects_bad_json() {
        assert!(Classifier::load("not json").is_err());
    }

    #[test]
    fn class_index_maps() {
        let c = Classifier::load(W).unwrap();
        assert_eq!(c.class_index("incomplete"), 1);
        assert_eq!(c.class_index("pore"), 2);
        assert_eq!(c.class_index("nope"), 0); // defaults to 0
    }
}

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct Model {
    pub classes: Vec<String>,
    pub weights: Vec<Vec<f32>>, // classes x features
    pub bias: Vec<f32>,
}

pub struct Classifier {
    m: Model,
}

impl Classifier {
    pub fn load(json: &str) -> Result<Self, serde_json::Error> {
        Ok(Classifier { m: serde_json::from_str(json)? })
    }

    /// Multinomial logistic (softmax) prediction over the feature vector.
    pub fn predict(&self, f: &[f32]) -> (String, f32) {
        let n = self.m.classes.len();
        let mut scores = vec![0.0f32; n];
        for (c, s) in scores.iter_mut().enumerate() {
            let mut acc = self.m.bias[c];
            for (j, &wv) in self.m.weights[c].iter().enumerate() {
                acc += wv * f.get(j).copied().unwrap_or(0.0);
            }
            *s = acc;
        }
        let mut smax = f32::MIN;
        for s in &scores { smax = smax.max(*s); }
        let mut denom = 0.0;
        for s in &scores { denom += (s - smax).exp(); }
        let mut best = 0;
        for (i, s) in scores.iter().enumerate() {
            if s > &scores[best] { best = i; }
        }
        let conf = (scores[best] - smax).exp() / denom;
        (self.m.classes[best].clone(), conf)
    }

    pub fn class_index(&self, name: &str) -> u8 {
        self.m.classes.iter().position(|c| c == name).unwrap_or(0) as u8
    }
}
