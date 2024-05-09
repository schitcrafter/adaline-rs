#![allow(dead_code)]

use std::error::Error;

use serde::Deserialize;

use crate::adaline::Adaline;

mod adaline;
mod plotting;

#[derive(Deserialize, Debug)]
struct Classification {
    /// true: 1, false: -1
    classification: bool,
    x: Vec<f32>,
}

const NUM_STEPS: u32 = 50;
const THRESHOLD: f32 = 0.98;

fn main() -> Result<(), Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("./data/adaline.csv")?;

    let records: Vec<_> = reader
        .deserialize()
        .filter_map(|res: Result<Vec<f32>, _>| {
            res.ok().map(|record: Vec<f32>| Classification {
                classification: record[0].is_sign_positive(),
                x: record[1..].into(),
            })
        })
        .collect();

    println!("Number of records: {}", records.len());

    plotting::plot_records(&records, "input-plot.png")?;

    let mut adaline = Adaline::new_random(0.002, records[0].x.len());

    println!("Weights: {:?}", adaline.weights());
    println!("Training starting");

    // train_on_one_record(&records, &mut adaline);

    println!("Training until model correctly classifies {}% of records correctly", THRESHOLD * 100.0);
    let correct_class_percents = train_on_all_records_until(&mut adaline, &records, THRESHOLD);
    println!("Took {} generations", correct_class_percents.len());
    
    plotting::plot_percentage_history(&correct_class_percents, THRESHOLD)?;

    println!("Done with training");

    println!(
        "{}% of records were correctly classified",
        adaline.classify_multiple_percent(&records) * 100.0
    );
    println!("Weights: {:?}", adaline.weights());

    Ok(())
}

fn train_on_all_records(adaline: &mut Adaline, records: &[Classification]) -> Vec<f32> {
    let mut correct_class_percents: Vec<f32> = Vec::new();
    for _ in 0..NUM_STEPS {
        correct_class_percents.push(adaline.classify_multiple_percent(records));
        for record in records {
            adaline.train(record);
        }
    }
    correct_class_percents
}

fn train_on_all_records_until(
    adaline: &mut Adaline,
    records: &Vec<Classification>,
    threshold: f32,
) -> Vec<f32> {
    let mut correct_class_percents: Vec<f32> = Vec::new();

    loop {
        let classification_percent = adaline.classify_multiple_percent(records);
        correct_class_percents.push(classification_percent);

        if classification_percent >= threshold {
            break;
        }

        if correct_class_percents.len() >= 10_000 {
            println!("ERROR: Stopping training after {} generations", correct_class_percents.len());
            break;
        }

        for record in records {
            adaline.train(record);
        }
    }

    correct_class_percents
}

fn train_on_one_record(records: &[Classification], adaline: &mut Adaline) {
    let test_record = records
        .iter()
        .find(|record| adaline.classify(&record.x) != record.classification)
        .unwrap();

    println!(
        "Adaline guesses correctly: {}",
        adaline.classify(&test_record.x) == test_record.classification
    );
    // let test_record = &records[1];
    let mut last_guess_correct = false;
    for i in 0..NUM_STEPS {
        let this_guess_correct = adaline.classify(&test_record.x) == test_record.classification;

        if !last_guess_correct && this_guess_correct {
            println!("Classification is now correct! (i={i})");
        } else if last_guess_correct && !this_guess_correct {
            println!("Classification is now incorrect again  (i={i})");
        }

        last_guess_correct = this_guess_correct;

        adaline.train(test_record);
    }
}
