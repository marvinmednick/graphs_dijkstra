//use std::env;
//use std::process; 
//use std::io::{self, Write}; // use std::error::Error;
//use std::cmp;
use log::{ info , error ,debug, warn,trace };
use std::path::Path;
use std::fs::File;
use std::io::{prelude::*, BufReader};
use std::collections::{HashMap,BTreeMap};
//use std::thread;
use regex::Regex;
//use std::fmt;

extern crate minheap;
use minheap::MinHeap;
mod cmd_line;
use crate::cmd_line::CommandArgs;




static mut MAX_OUT_LEVEL : u32= 0;
static mut MAX_IN_LEVEL : u32 = 0;

#[derive(Debug, Clone)]
struct Vertex {
	vertex_id: usize,
    // list of incoming edges along with a count for duplicates
	incoming: BTreeMap<Edge,usize>,
    // total count of incoming edges
	incoming_cnt: usize,
    // list of incoming edges along with a count for duplicates
	outgoing: BTreeMap<Edge,usize>,
    // total count of outgoing edges
	outgoing_cnt: usize,
}

#[derive(Debug,Clone,Ord,PartialOrd,Eq,PartialEq)]
struct Edge {
    vertex: usize,
    weight: u32
}

impl Vertex {

	pub fn new(id : &usize) -> Vertex {
		let incoming = BTreeMap::<Edge,usize>::new();
		let outgoing = BTreeMap::<Edge,usize>::new();
		Vertex {vertex_id: id.clone(), 
				incoming: incoming, 
				outgoing: outgoing,
				incoming_cnt : 0,
				outgoing_cnt : 0,
				}
	}
	
	pub fn add_outgoing(&mut self, vertex_id: usize, weight: u32) {
        let edge = Edge {vertex: vertex_id, weight: weight };
		let counter = self.outgoing.entry(edge).or_insert(0);
		*counter += 1;
		self.outgoing_cnt += 1;
	}

	pub fn del_outgoing (&mut self, vertex_id: usize, weight: u32) ->  Result <(), String> {

        let edge = Edge {vertex: vertex_id, weight: weight };

		match self.outgoing.get_mut(&edge) {
			None | Some(0)  => Err("Invalid Vertex".to_string()),
			Some(1)        =>  	{ 	
									self.outgoing.remove(&edge); 
									self.outgoing_cnt -= 1;
									Ok(())
								}, 
			Some(x)        => 	{	*x -=1;  
								 	self.outgoing_cnt -= 1;
								 	Ok(())
								},
		}
	}

	pub fn add_incoming(&mut self, vertex_id: usize, weight: u32) {
        let edge = Edge {vertex: vertex_id, weight: weight };
		let counter = self.incoming.entry(edge).or_insert(0);
		*counter += 1;
		self.incoming_cnt += 1;
	}

	pub fn del_incoming (&mut self, vertex_id: usize, weight: u32) -> Result<(),String> {
	
        let edge = Edge {vertex: vertex_id, weight: weight };
		match self.incoming.get_mut(&edge) {
			None | Some(0)  => Err("Invalid Vertex".to_string()),
			Some(1)        =>	{ 
									self.incoming.remove(&edge); 
									self.incoming_cnt -= 1;
									Ok(())
								}, 
			Some(x)        => 	{
									*x -=1;
									self.incoming_cnt -= 1;
									Ok(())
								},
		}

	}
}


#[derive(Debug,Clone)]
struct Graph {
    // map from vertex Id to actual Vertex data
	vertex_map:  BTreeMap::<usize, Vertex>,
	edge_count:  usize,
    // map indicating if vertex id has been explored
	explored:  HashMap::<usize,bool>,
    // list of vertex ids in order finished processing
	pub finished_order:  Vec::<usize>,
	pub start_search:  HashMap::<usize,Vec::<usize>>,
	top_search_cnts:  HashMap::<usize, usize>,
    // heap of the unprocessed vertexs by  current score
    pub unprocessed_vertex : MinHeap::<u32>,
    // map for all processed vertex from vertex id to its current score
    pub processed_vertex : HashMap::<usize,u32>,
}


impl Graph {
	pub fn new() -> Graph {
		let v_map = BTreeMap::<usize, Vertex>::new();
		Graph {
				vertex_map: v_map,
				edge_count: 0,
				explored:  HashMap::<usize,bool>::new(),
				finished_order:  Vec::<usize>::new(),
				start_search : HashMap::<usize,Vec::<usize>>::new(),
				top_search_cnts : HashMap::<usize,usize>::new(),
                unprocessed_vertex : MinHeap::<u32>::new(),
                processed_vertex : HashMap::<usize,u32>::new(),
		}
	}


	pub fn get_outgoing(&self, vertex: usize) -> Vec<Edge>{
		let v = self.vertex_map.get(&vertex).unwrap();
		v.outgoing.keys().cloned().collect()
		
	}

	pub fn get_incoming(&self,vertex: usize) -> Vec<Edge> {
		let v = self.vertex_map.get(&vertex).unwrap();
		v.incoming.keys().cloned().collect()
		
	}


	pub fn get_vertexes(&self) -> Vec<usize> {
		self.vertex_map.keys().cloned().collect()
			
	}

	pub fn print_vertexes(&self) {
		for (key, value) in &self.vertex_map {
			let out_list : String = value.outgoing.iter().map(|(x, y)| if y > &1 {format!("{:?}({}) ; ",x,y) } else { format!("{:?} ;",x)}).collect();
			println!("Vertex {} ({}) :  {}",key,value.vertex_id,out_list);
            println!("       key {:?}   value {:?}", key, value);
		}
					
	}

	pub fn create_vertex(&mut self,id: &usize) -> Option<usize> {

		if self.vertex_map.contains_key(&id) {
			None
		} 
		else { 
			let v = Vertex::new(&id);
			self.vertex_map.insert(id.clone(),v.clone());
			Some(self.vertex_map.len())  
		}
	}

	pub fn add_search_entry(&mut self, vertex: usize, count: usize) {

			self.top_search_cnts.insert(vertex,count);
			let mut removed = None;
			if self.top_search_cnts.len() > 10 {
				let top_search_iter = self.top_search_cnts.iter();
				let mut top_search_count_vec : Vec::<(usize, usize)> = top_search_iter.map(|(k,v)| (*k, *v)).collect();
				top_search_count_vec.sort_by(|a, b| b.1.cmp(&a.1));
				removed = top_search_count_vec.pop();
			}
			if let Some(entry) = removed {
				self.top_search_cnts.remove(&entry.0);
				
			}
			
	}

	pub fn dfs_outgoing(&mut self, vertex_id:  usize, start_vertex: usize, level: u32) {
			
        let spacer = (0..level*5).map(|_| " ").collect::<String>();
        unsafe {
			if level > MAX_OUT_LEVEL {
				MAX_OUT_LEVEL = level;
					warn!("reached level {}", MAX_OUT_LEVEL);
			}
        }
			
        // Set current node to explored
        self.explored.insert(vertex_id,true);

        let cur_len: usize;
    
        {
            let group_list = self.start_search.entry(start_vertex).or_insert(Vec::<usize>::new());
            group_list.push(vertex_id);
            cur_len = group_list.len();
        }
        self.add_search_entry(start_vertex,cur_len);

        
        let next_v : Vertex;

        if let Some(vertex) = self.vertex_map.get(&vertex_id) {

            next_v = vertex.clone();
        }

        else {
            panic!("invalid vertex");
        }

        // Search through each edge
        for edge in next_v.outgoing.keys() {
            let next_vertex = edge.vertex.clone();
            if !self.explored.contains_key(&edge.vertex) {
                self.dfs_outgoing(next_vertex,start_vertex,level+1);
            }
            else {
                trace!("{}Vertex {} is already explored",spacer,edge.vertex);
            }
        }
        // so add it to the finished list
        self.finished_order.push(vertex_id);
	}

	pub fn dfs_incoming(&mut self, vertex_id:  usize, start_vertex: usize, level: u32) {
			
        let spacer = (0..level*5).map(|_| " ").collect::<String>();
        unsafe {
			if level > MAX_IN_LEVEL {
				MAX_IN_LEVEL = level;
				warn!("reached level {}", MAX_IN_LEVEL);
			}
        }
			
        // Set current node to explored
        self.explored.insert(vertex_id,true);

        let group_list = self.start_search.entry(start_vertex).or_insert(Vec::<usize>::new());
        group_list.push(vertex_id);
        let cur_len = group_list.len();
        self.add_search_entry(start_vertex,cur_len);

        let next_v : Vertex;

        if let Some(vertex) = self.vertex_map.get(&vertex_id) {

            next_v = vertex.clone();
        }

        else {
            panic!("invalid vertex");
        }

        // Search through each edge
        for edge in next_v.incoming.keys() {
            let next_vertex = edge.vertex.clone();
            if !self.explored.contains_key(&edge.vertex) {
                self.dfs_incoming(next_vertex,start_vertex,level+1);
            }
        else {
            trace!("{}Vertex {} is already explored",spacer,edge.vertex);
        }
    }
    // so add it to the finished list
    self.finished_order.push(vertex_id);
}

pub fn dfs_loop_incoming(&mut self, list: &Vec<usize>) {

		debug!("Looping on incoming DFS");
		self.finished_order = Vec::<usize>::new();
		self.start_search = HashMap::<usize,Vec::<usize>>::new();
		self.explored = HashMap::<usize,bool>::new();
		self.top_search_cnts = HashMap::<usize,usize>::new();

		let mut _count : usize = 0;
		for v in list {
/*			if _count % 1000000 == 0 {
				print!("*");
				io::stdout().flush().unwrap();
			} */
			let vertex = v.clone();
			info!("Looping on {}",vertex);
			if !self.explored.contains_key(&vertex) {
				self.dfs_incoming(vertex,vertex,0);
			}
			_count += 1;
		}
	}

	pub fn dfs_loop_outgoing(&mut self, list: &Vec<usize>) {
		info!("Looping on outgoing DFS");
		self.finished_order = Vec::<usize>::new();
		self.start_search = HashMap::<usize,Vec::<usize>>::new();
		self.explored = HashMap::<usize,bool>::new();
		self.top_search_cnts = HashMap::<usize,usize>::new();

		let mut _count : usize = 0;
		for v in list {
/*			if _count % 1000000 == 0 {
				print!("#");
				io::stdout().flush().unwrap();
			} */
			let vertex = v.clone();
			debug!("Looping on {}",vertex);
			if !self.explored.contains_key(&vertex) {
				self.dfs_outgoing(vertex,vertex,0);
			}
		}
	}

	pub fn add_edge(&mut self, v1: usize, v2: usize, weight: u32) -> Option<usize> {

		//create the vertexes, if the don't exist
		self.create_vertex(&v1);
		self.create_vertex(&v2);

		let v_map = &mut self.vertex_map;
		// add the edge to the first vertex's adjanceny list
		let vert = v_map.get_mut(&v1).unwrap(); 
		vert.add_outgoing(v2,weight);
		let new_cnt = vert.outgoing_cnt.clone();

		// add the edge to the second vertex adjacentcy list
		let vert2 = v_map.get_mut(&v2).unwrap(); 
		vert2.add_incoming(v1,weight);

		self.edge_count += 1;
		Some(new_cnt)

	}

	pub fn delete_edge(&mut self,v1 : usize, v2 : usize, weight: u32) -> Result<(),String>  {
	
		self.vertex_map.get_mut(&v1).unwrap().del_outgoing(v2,weight)?	;
		self.vertex_map.get_mut(&v2).unwrap().del_incoming(v1,weight)?;
		self.edge_count -= 1;
		Ok(())

	}


    // dijkstra shortest path
    pub fn update_scoring(&mut self, id: usize) {
        debug!("Dijsktra scoring {}",id);
        let adj_vertexes = self.get_outgoing(id);
        
        // get the distance/score from the current vertex as the base
        let cur_vertex_distance = self.processed_vertex.get(&id).unwrap().clone();

        // update each of this nodes adjancent vertexes, if the new distance
        // is < the current distance
        for v in adj_vertexes {
            debug!("Dijsktra updating adjacent {:?}",v);
            // if the adjacent vertex is still in the unprocessed list, then 
            // update the scoring, otherwise skip it (since its already in the processed list)
            if let Some(cur_score) = self.unprocessed_vertex.peek_id_data(v.vertex) {
                let new_score = cur_vertex_distance + v.weight;
                if new_score < cur_score {
                    debug!("Update scoring on {} from {} to {}",v.vertex,cur_score,new_score);
                    let vertex_index = self.unprocessed_vertex.get_id_index(v.vertex).unwrap().clone();
                    self.unprocessed_vertex.update(vertex_index,new_score);
                    trace!("Unprocessed: {:?}",self.unprocessed_vertex)
                }
             }       
                
            
        }

    }

    pub fn shortest_paths(&mut self, starting_vertex: usize) {
        info!("Starting shortest path with {}",starting_vertex);

        if let Some(starting_index) = self.unprocessed_vertex.get_id_index(starting_vertex) {

            let index = starting_index.clone();
            self.unprocessed_vertex.delete(index);
            
            // setup the initial distance for the starting vertex to 0 (to itself)
            self.processed_vertex.insert(starting_vertex,0);

            self.update_scoring(starting_vertex);

            while let Some((next_vertex,next_vertex_score)) = self.unprocessed_vertex.get_min_entry() {
                debug!("Processing vertex {} score: {}",next_vertex,next_vertex_score);
                self.processed_vertex.insert(next_vertex,next_vertex_score);
                self.update_scoring(next_vertex);
            }
         }       
        else {
            error!("Starting vertex {} is not in the graph",starting_vertex);
        }

    }

            

    

}

fn main() {


    env_logger::init();
    let cmd_line = CommandArgs::new();

    info!("Command line is: {:?}!",cmd_line);



    info!("Calulating shortest path from Vertex {} to all other vertexes",cmd_line.start_vertex);
  // Create a path to the desired file
    let path = Path::new(&cmd_line.filename);
    let display = path.display();


    // Open the path in read-only mode, returns `io::Result<File>`
    let file = match File::open(&path) {
        Err(why) => panic!("couldn't open {}: {}", display, why),
        Ok(file) => file,
    };

    let reader = BufReader::new(file);

	let mut g = Graph::new();

	let mut _count = 0;
    for line in reader.lines() {
		_count += 1;	
		let line_data = line.unwrap();

        // split the line into the vertex and the list of adjacent vertexes/weight pairs
        let re_vertex = Regex::new(r"\s*(?P<vertex>\d+)\s+(?P<adjacent_list>.*$)").unwrap();
        // adjacent vertexes are in the format vertex,weight   - and regex below allows for
        // whitespace
        let caps = re_vertex.captures(&line_data).unwrap();
        let text1 = caps.get(1).map_or("", |m| m.as_str());
        let vertex = text1.parse::<usize>().unwrap();
        debug!("Reading connectsion for vertex {}",vertex);

        let re_adjacent = Regex::new(r"\s*(?P<vertex>\d+)\s*,\s*(?P<weight>\d*)").unwrap();
        let text2 = caps.get(2).map_or("", |m| m.as_str());
        trace!("Adjacency info: {}",text2);

        let mut count =0;
        for caps in re_adjacent.captures_iter(text2) {
            let dest_vertex = caps["vertex"].parse::<usize>().unwrap(); 
            let weight = caps["weight"].parse::<u32>().unwrap(); 
            debug!("Adding connection from {} to {} with weight {}",vertex,dest_vertex,weight);
			let _num_edges = g.add_edge(vertex,dest_vertex,weight);
            count += 1;

        }
        info!("Vertex:  {} - {} edges",vertex,count);
        g.unprocessed_vertex.insert(vertex,100000000);
    }

    g.shortest_paths(cmd_line.start_vertex);

    if cmd_line.display_dest.len() > 0 {
        let mut is_first = true;
        for v in cmd_line.display_dest {
            if !is_first{
                print!(",");
            }
            is_first = false;
            if g.processed_vertex.contains_key(&v) {
                print_vertex_result(&v, g.processed_vertex.get(&v).unwrap(),cmd_line.short_disp);
            }
            else {
                error!("Dest Vertex {} is invalid",v);
            }
        }
        println!();

    }
    else {
        for v in g.vertex_map.keys() {
            print_vertex_result(v, g.processed_vertex.get(&v).unwrap(),cmd_line.short_disp);
        }
        println!();
    }

}

fn print_vertex_result(vertex: &usize, result: &u32, short: bool) {

    if short {
        print!("{}", result);
    }
    else {
        println!("v {} - {}", vertex, result);
    }

}



/*
 * the rest of this file sets up unit tests
 * to run these, the command will be:
 * cargo test --package rust-template -- --nocapture
 * Note: 'rust-template' comes from Cargo.toml's 'name' key
 */
/*
// use the attribute below for unit tests
#[cfg(test)]
mod tests {
    use super::*;

	fn setup_basic1() -> Graph {
		let mut g = Graph::new();
		assert_eq!(g.add_edge(1,2),Some(1));
		assert_eq!(g.add_edge(1,3),Some(2));
		assert_eq!(g.add_edge(2,3),Some(1));
		assert_eq!(g.add_edge(2,4),Some(2));
		assert_eq!(g.add_edge(3,4),Some(1));
		assert_eq!(g.get_outgoing(1),&[2,3]);
		assert_eq!(g.get_outgoing(2),&[3,4]);
		assert_eq!(g.get_outgoing(3),&[4]);
		assert_eq!(g.get_outgoing(4),&[]);
		g
	} 

    #[test]
    fn basic() {
		let mut g = Graph::new();
		assert_eq!(g.create_vertex(&1),Some(1));
		assert_eq!(g.create_vertex(&2),Some(2));
		assert_eq!(g.add_edge(1,2),Some(1));
		assert_eq!(g.get_vertexes(),vec!(1,2));
		assert_eq!(g.create_vertex(&3),Some(3));
		assert_eq!(g.add_edge(1,3),Some(2));
		assert_eq!(g.add_edge(2,3),Some(1));
		assert_eq!(g.get_vertexes(),vec!(1,2,3));
		assert_eq!(g.add_edge(1,4),Some(3));
		assert_eq!(g.get_vertexes(),vec!(1,2,3,4));
		println!("{:?}",g);

    }

	#[test]
	fn test_add() {
		let mut g = Graph::new();
		assert_eq!(g.add_edge(1,2),Some(1));
		assert_eq!(g.get_outgoing(1),&[2]);
		assert_eq!(g.get_incoming(2),&[1]);
		assert_eq!(g.add_edge(1,3),Some(2));
		assert_eq!(g.get_outgoing(1),&[2,3]);
		assert_eq!(g.get_incoming(2),&[1]);
	}

	#[test]
	fn test_add_del() {
		let mut g = setup_basic1();
		assert_eq!(g.get_outgoing(1),&[2,3]);
		assert_eq!(g.add_edge(1,2),Some(3));
		assert_eq!(g.get_outgoing(1),&[2,3]);
		assert_eq!(g.get_outgoing(2),&[3,4]);
		assert_eq!(g.get_outgoing(3),&[4]);
		assert_eq!(g.delete_edge(1,2),Ok(()));
		assert_eq!(g.get_outgoing(1),&[2,3]);
		assert_eq!(g.delete_edge(1,2),Ok(()));
		assert_eq!(g.get_outgoing(1),&[3]);
		
	}


 }
 */
