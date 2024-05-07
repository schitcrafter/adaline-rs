use std::error::Error;

use plotters::prelude::*;

pub fn plot_percentage_history(history: &[f32], threshold: f32) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("history-plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("% of correct classifications", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..(history.len() as f32), 0f32..100.0)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            history.iter()
                .enumerate()
                .map(|(i, percent)| (i as f32, percent * 100.0)),
            &BLUE,
        ))?
        .label("Percentage of correct classifications")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.draw_series(LineSeries::new(
        [(0.0, threshold * 100.0), ((history.len() as f32), threshold * 100.0)],
        &RED
    ))?
    .label("Threshold");
    // .legend()

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}