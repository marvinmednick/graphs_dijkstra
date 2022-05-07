extern crate clap;

use clap::{Arg, Command};

#[derive(Debug)]
pub struct CommandArgs  {
    pub filename: String,
    pub start_vertex: u32,
    pub display_dest: Vec::<u32>,
}

impl CommandArgs  {
    pub fn new() -> Self {
        // basic app information
        let app = Command::new("dijkstra")
            .version("1.0")
            .about("Says hello")
            .author("Marvin Mednick");

        // Define the name command line option
        let filename_option = Arg::new("file")
            .takes_value(true)
            .help("Input file name")
            .required(true);

        let starting_option = Arg::new("start")
            .takes_value(true)
            .help("Starting Vertex")
            .required(true);

        let display_option = Arg::new("display")
            .help("Starting Vertex")
            .multiple_values(true);

        // now add in the argument we want to parse
        let mut app = app.arg(filename_option);
        app = app.arg(starting_option);
        app = app.arg(display_option);

        // extract the matches
        let matches = app.get_matches();

        // Extract the actual name
        let filename = matches.value_of("file")
            .expect("Filename can't be None, we said it was required");

        let num_str = matches.value_of("start");

        let start = match num_str {
            None => { println!("Start is None..."); 0},
            Some(s) => {
                match s.parse::<u32>() {
                    Ok(n) => n,
                    Err(_) => {println!("That's not a number! {}", s); 0},
                }
            }
        };
        let disp_vertex: Vec<_> = matches.values_of("display")
                                    .unwrap_or_default()
                                    .map(|s| s.parse().expect("parse error"))
                                    .collect();

        println!("clap args: {} {} {:?}",filename, start,disp_vertex);

        CommandArgs { filename: filename.to_string(), start_vertex : start, display_dest: disp_vertex}
    }   
}
