//! Run vision-metrology's textbook calipers on a CaliperBench requests file.
//!
//! CaliperBench runs a black-box implementation as
//! `COMMAND [ARGS...] --requests R --data-root D --output O` and expects one prediction
//! row per request (its JSONL protocol). This is that command:
//!
//! ```text
//! caliperbench_run --method {gradient_parabolic|gradient_integer|gradient_gaussian|
//!                            midpoint_crossing|half_contrast}
//!                  [--operator central|dog] [--params FILE] [--trace FILE]
//!                  --requests R --data-root D --output O
//! ```
//!
//! `gradient_parabolic`, `gradient_integer` and `midpoint_crossing` reproduce
//! CaliperBench's baselines of the same names; `--params` takes the same JSON as
//! `caliperbench run --params`. `--trace` writes each request's intermediates (profile,
//! smoothed profile, gradient and candidates, or end levels) as JSONL, with the keys of
//! CaliperBench's lab trace.
//!
//! The exit status is 0 once the output is written, failed rows included, and non-zero
//! only for bad arguments, bad parameters or an unreadable requests file.
//!
//! ## Run
//! ```text
//! cargo build --release -p vision-metrology --example caliperbench_run
//! uv run caliperbench run-external outputs/requests.jsonl --data-root data \
//!   --output outputs/vm.jsonl --label vm-parabolic -- \
//!   /path/to/target/release/examples/caliperbench_run --method gradient_parabolic
//! ```

#[path = "common/caliperbench.rs"]
mod caliperbench;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use clap::builder::PossibleValuesParser;

use caliperbench::{Method, Operator, Params, RunOptions, run};

#[derive(Parser)]
#[command(about = "Run vision-metrology calipers on CaliperBench requests (JSONL protocol)")]
struct Cli {
    /// Edge-location method.
    #[arg(long, value_parser = PossibleValuesParser::new(Method::ALL.map(|(name, _)| name)))]
    method: String,
    /// Derivative operator: a Gaussian then central differences, or derivative of Gaussian.
    #[arg(long, default_value = "central", value_parser = PossibleValuesParser::new(["central", "dog"]))]
    operator: String,
    /// JSON object of baseline parameters (CaliperBench's `--params` format).
    #[arg(long)]
    params: Option<PathBuf>,
    /// Write each request's intermediates here, as JSONL.
    #[arg(long)]
    trace: Option<PathBuf>,
    /// Requests file (JSONL).
    #[arg(long)]
    requests: PathBuf,
    /// Directory the requests' image paths are relative to.
    #[arg(long)]
    data_root: PathBuf,
    /// Predictions file to write (JSONL).
    #[arg(long)]
    output: PathBuf,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let params = match &cli.params {
        None => Ok(Params::default()),
        Some(path) => std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))
            .and_then(|text| Params::from_json(&text)),
    };
    let params = match params {
        Ok(params) => params,
        Err(e) => {
            eprintln!("caliperbench_run: {e}");
            return ExitCode::from(2);
        }
    };
    let opts = RunOptions {
        method: Method::parse(&cli.method).expect("clap checked the method"),
        operator: Operator::parse(&cli.operator).expect("clap checked the operator"),
        params,
        requests: cli.requests,
        data_root: cli.data_root,
        output: cli.output,
        trace: cli.trace,
    };
    match run(&opts) {
        Ok(summary) => {
            eprintln!(
                "caliperbench_run: {} ok, {} failed -> {}",
                summary.ok,
                summary.failed,
                opts.output.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("caliperbench_run: {e}");
            ExitCode::from(2)
        }
    }
}
