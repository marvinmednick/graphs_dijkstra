extern crate clap;

use clap::{Arg, Command};

#[derive(Debug)]
pub struct CommandArgs  {
    pub filename: String,
    pub start_vertex: usize,
    pub display_dest: Vec::<usize>,
    pub short_disp:  bool,
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

        let short_display_option = Arg::new("short_disp")
            .long("short_disp")
            .short('1')
            .takes_value(false)
            .help("Short display: show final weights on one line");

        let display_option = Arg::new("display")
            .help("Starting Vertex")
            .multiple_values(true);

        // now add in the argument we want to parse
        let mut app = app.arg(filename_option)
                .arg(starting_option)
                .arg(display_option)
                .arg(short_display_option);

        // extract the matches
        let matches = app.get_matches();

        // Extract the actual name
        let filename = matches.value_of("file")
            .expect("Filename can't be None, we said it was required");

        let num_str = matches.value_of("start");

        let start = match num_str {
            None => { println!("Start is None..."); 0},
            Some(s) => {
                match s.parse::<usize>() {
                    Ok(n) => n,
                    Err(_) => {println!("That's not a number! {}", s); 0},
                }
            }
        };
        let disp_vertex: Vec<_> = matches.values_of("display")
                                    .unwrap_or_default()
                                    .map(|s| s.parse().expect("parse error"))
                                    .collect();


        let short_disp = matches.is_present("short_disp");

//        println!("clap args: {} {} {:?}",filename, start,disp_vertex);

        CommandArgs { filename: filename.to_string(), start_vertex : start, display_dest: disp_vertex, short_disp: short_disp}
    }   
}
