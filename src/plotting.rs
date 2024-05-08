use std::error::Error;

use plotly::{
    common::Mode,
    Plot, Scatter,
};

use crate::Classification;

pub fn plot_percentage_history(history: &[f32], threshold: f32) -> Result<(), Box<dyn Error>> {
    let mut plot = Plot::new();

    let history_trace = Scatter::new(
        (0..history.len()).map(|i| i as f32).collect(),
        history.to_vec(),
    )
    .fill_color("blue")
    .name("History");

    let threshold_trace =
        Scatter::new(vec![0f32, history.len() as f32], vec![threshold, threshold])
            .fill_color("red")
            .name("Threshold");

    plot.add_trace(history_trace);
    plot.add_trace(threshold_trace);

    plot.write_html("percentage_history.html");

    Ok(())
}

pub fn plot_records(records: &[Classification], path: &str) -> Result<(), Box<dyn Error>> {
    let mut plot = Plot::new();

    let (records_true, records_false): (Vec<_>, Vec<_>) =
        records.iter().partition(|class| class.classification);

    let (x, y): (Vec<_>, Vec<_>) = records_false
        .iter()
        .map(|class| (class.x[0], class.x[1]))
        .unzip();

    let records_trace_false = Scatter::new(x, y)
        .name("Records with classification -1")
        .mode(Mode::Markers)
        .fill_color("blue");

    let (x, y): (Vec<_>, Vec<_>) = records_true
        .iter()
        .map(|class| (class.x[0], class.x[1]))
        .unzip();

    let records_trace_true = Scatter::new(x, y)
        .name("Records with classification 1")
        .mode(Mode::Markers)
        .fill_color("red");

    plot.add_trace(records_trace_false);
    plot.add_trace(records_trace_true);

    plot.write_html(path);

    Ok(())
}
