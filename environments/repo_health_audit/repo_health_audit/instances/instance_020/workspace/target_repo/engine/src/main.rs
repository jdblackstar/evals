use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: engine <input> <output>");
        std::process::exit(1);
    }

    let input = &args[1];
    let output = &args[2];

    let data = fs::read_to_string(input).expect("failed to read input");
    let processed = data.to_uppercase();
    fs::write(output, processed).expect("failed to write output");

    println!("Processing complete");
}
