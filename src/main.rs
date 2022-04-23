use std::env; use std::process; use std::io::{self, Write}; // use std::error::Error;
//use std::cmp;
use std::path::Path;
use std::fs::File;
use std::io::{prelude::*, BufReader};
use std::collections::{HashMap,BTreeMap};
use std::thread;

static mut MAX_OUT_LEVEL : u32= 0;
static mut MAX_IN_LEVEL : u32 = 0;

#[derive(Debug, Clone)]
struct Vertex {
	vertex_id: u32,
	incoming: BTreeMap<u32,u32>,
	incoming_cnt: usize,
	outgoing: BTreeMap<u32,u32>,
	outgoing_cnt: usize,
}

impl Vertex {

	pub fn new(id : &u32) -> Vertex {
		let incoming = BTreeMap::<u32,u32>::new();
		let outgoing = BTreeMap::<u32,u32>::new();
		Vertex {vertex_id: id.clone(), 
				incoming: incoming, 
				outgoing: outgoing,
				incoming_cnt : 0,
				outgoing_cnt : 0,
				}
	}
	
	pub fn add_outgoing(&mut self, vertex_id: u32) {
		let counter = self.outgoing.entry(vertex_id).or_insert(0);
		*counter += 1;
		self.outgoing_cnt += 1;
	}

	pub fn del_outgoing (&mut self, vertex_id: u32) ->  Result <(), String> {

		match self.outgoing.get_mut(&vertex_id) {
			None | Some(0)  => Err("Invalid Vertex".to_string()),
			Some(1)        =>  	{ 	
									self.outgoing.remove(&vertex_id); 
									self.outgoing_cnt -= 1;
									Ok(())
								}, 
			Some(x)        => 	{	*x -=1;  
								 	self.outgoing_cnt -= 1;
								 	Ok(())
								},
		}
	}

	pub fn add_incoming(&mut self, vertex_id: u32) {
		let counter = self.incoming.entry(vertex_id).or_insert(0);
		*counter += 1;
		self.incoming_cnt += 1;
	}

	pub fn del_incoming (&mut self, vertex_id: u32) -> Result<(),String> {
	
		match self.incoming.get_mut(&vertex_id) {
			None | Some(0)  => Err("Invalid Vertex".to_string()),
			Some(1)        =>	{ 
									self.incoming.remove(&vertex_id); 
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
	vertex_map:  BTreeMap::<u32, Vertex>,
	edge_count:  u32,
	explored:  HashMap::<u32,bool>,
	pub finished_order:  Vec::<u32>,
	pub start_search:  HashMap::<u32,Vec::<u32>>,
	top_search_cnts:  HashMap::<u32, usize>,
}


impl Graph {
	pub fn new() -> Graph {
		let v_map = BTreeMap::<u32, Vertex>::new();
		Graph {
				vertex_map: v_map,
				edge_count: 0,
				explored:  HashMap::<u32,bool>::new(),
				finished_order:  Vec::<u32>::new(),
				start_search : HashMap::<u32,Vec::<u32>>::new(),
				top_search_cnts : HashMap::<u32,usize>::new(),
		}
	}


	pub fn get_outgoing(&self, vertex: u32) -> Vec<u32>{
		let v = self.vertex_map.get(&vertex).unwrap();
		v.outgoing.keys().cloned().collect()
		
	}

	pub fn get_incoming(&self,vertex: u32) -> Vec<u32> {
		let v = self.vertex_map.get(&vertex).unwrap();
		v.incoming.keys().cloned().collect()
		
	}


	pub fn get_vertexes(&self) -> Vec<u32> {
		self.vertex_map.keys().cloned().collect()
			
	}

	pub fn print_vertexes(&self) {
		for (key, value) in &self.vertex_map {
			let out_list : String = value.outgoing.iter().map(|(x, y)| if y > &1 {format!("{}({}) ; ",x,y) } else { format!("{} ;",x)}).collect();
			println!("Vertex {} ({}) :  {}",key,value.vertex_id,out_list);
		}
					
	}

	pub fn create_vertex(&mut self,id: &u32) -> Option<usize> {

		if self.vertex_map.contains_key(&id) {
			None
		} 
		else { 
			let v = Vertex::new(&id);
			self.vertex_map.insert(id.clone(),v.clone());
			Some(self.vertex_map.len())  
		}
	}

	pub fn add_search_entry(&mut self, vertex: u32, count: usize) {

			self.top_search_cnts.insert(vertex,count);
			let mut removed = None;
			if self.top_search_cnts.len() > 10 {
				let top_search_iter = self.top_search_cnts.iter();
				let mut top_search_count_vec : Vec::<(u32, usize)> = top_search_iter.map(|(k,v)| (*k, *v)).collect();
				top_search_count_vec.sort_by(|a, b| b.1.cmp(&a.1));
				removed = top_search_count_vec.pop();
			}
			if let Some(entry) = removed {
				self.top_search_cnts.remove(&entry.0);
				
			}
			
	}

	pub fn dfs_outgoing(&mut self, vertex_id:  u32, start_vertex: u32, level: u32) {
			
//			let spacer = (0..level*5).map(|_| " ").collect::<String>();
			unsafe {
			if level > MAX_OUT_LEVEL {
				MAX_OUT_LEVEL = level;
//					println!("reached level {}", MAX_OUT_LEVEL);
			}
			}
			
			// Set current node to explored
			self.explored.insert(vertex_id,true);

			let cur_len: usize;
		
			{
				let group_list = self.start_search.entry(start_vertex).or_insert(Vec::<u32>::new());
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
				let next_vertex = edge.clone();
				if !self.explored.contains_key(&edge) {
					self.dfs_outgoing(next_vertex,start_vertex,level+1);
				}
				else {
			//		println!("{}Vertex {} is already explored",spacer,edge);
				}
			}
			// so add it to the finished list
			self.finished_order.push(vertex_id);
	}

	pub fn dfs_incoming(&mut self, vertex_id:  u32, start_vertex: u32, level: u32) {
			
//			let spacer = (0..level*5).map(|_| " ").collect::<String>();
			unsafe {
			if level > MAX_IN_LEVEL {
				MAX_IN_LEVEL = level;
//				println!("reached level {}", MAX_IN_LEVEL);
			}
			}
			
			// Set current node to explored
			self.explored.insert(vertex_id,true);

			let group_list = self.start_search.entry(start_vertex).or_insert(Vec::<u32>::new());
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
				let next_vertex = edge.clone();
				if !self.explored.contains_key(&edge) {
					self.dfs_incoming(next_vertex,start_vertex,level+1);
				}
				else {
			//		println!("{}Vertex {} is already explored",spacer,edge);
				}
			}
			// so add it to the finished list
			self.finished_order.push(vertex_id);
	}

	pub fn dfs_loop_incoming(&mut self, list: &Vec<u32>) {

//		println!("Looping on incoming DFS");
		self.finished_order = Vec::<u32>::new();
		self.start_search = HashMap::<u32,Vec::<u32>>::new();
		self.explored = HashMap::<u32,bool>::new();
		self.top_search_cnts = HashMap::<u32,usize>::new();

		let mut _count : usize = 0;
		for v in list {
/*			if _count % 1000000 == 0 {
				print!("*");
				io::stdout().flush().unwrap();
			} */
			let vertex = v.clone();
//			println!("Looping on {}",vertex);
			if !self.explored.contains_key(&vertex) {
				self.dfs_incoming(vertex,vertex,0);
			}
			_count += 1;
		}
	}

	pub fn dfs_loop_outgoing(&mut self, list: &Vec<u32>) {
//		println!("Looping on outgoing DFS");
		self.finished_order = Vec::<u32>::new();
		self.start_search = HashMap::<u32,Vec::<u32>>::new();
		self.explored = HashMap::<u32,bool>::new();
		self.top_search_cnts = HashMap::<u32,usize>::new();

		let mut _count : usize = 0;
		for v in list {
/*			if _count % 1000000 == 0 {
				print!("#");
				io::stdout().flush().unwrap();
			} */
			let vertex = v.clone();
//			println!("Looping on {}",vertex);
			if !self.explored.contains_key(&vertex) {
				self.dfs_outgoing(vertex,vertex,0);
			}
		}
	}

/*			
	pub fn DFS2(&mut self, vertex_id:  u32, level: u32) {
			let spacer = (0..level*5).map(|_| " ").collect::<String>();
			
			println!("{}Exploring {}",spacer,vertex_id);
			// Set current node to explored
			self.explored.insert(vertex_id,true);


			if let Some(vertex) = self.vertex_map.get(&vertex_id) {
				println!("{}Vertex {:?}",spacer,vertex);
				println!("{}searching through {:?}",spacer,vertex.outgoing.keys());

				// Search through each edge
				for edge in vertex.outgoing.keys() {
					let next_vertex = edge.clone();
					if !self.explored.contains_key(&edge) {
						self.DFS(next_vertex,level+1);
					}
					else {
						println!("{}Vertex {} is already explored",spacer,edge);
					}
				}
				//Done with vertex (all outgoing edges explorered
				// so add it to the finished list
				self.finished_order.push(vertex_id);
			}

			else {
				panic!("invalid vertex");
			}

	}

*/

	pub fn add_edge(&mut self, v1: u32, v2: u32) -> Option<usize> {

		//create the vertexes, if the don't exist
		self.create_vertex(&v1);
		self.create_vertex(&v2);

		let v_map = &mut self.vertex_map;
		// add the edge to the first vertex's adjanceny list
		let vert = v_map.get_mut(&v1).unwrap(); 
		vert.add_outgoing(v2);
		let new_cnt = vert.outgoing_cnt.clone();

		// add the edge to the second vertex adjacentcy list
		let vert2 = v_map.get_mut(&v2).unwrap(); 
		vert2.add_incoming(v1);

		self.edge_count += 1;
		Some(new_cnt)

	}

	pub fn delete_edge(&mut self,v1 : u32, v2 : u32) -> Result<(),String>  {
	
		self.vertex_map.get_mut(&v1).unwrap().del_outgoing(v2)?	;
		self.vertex_map.get_mut(&v2).unwrap().del_incoming(v1)?;
		self.edge_count -= 1;
		Ok(())

	}
}



fn main() {


    let args: Vec<String> = env::args().collect();

	println!("Args {:?} {}",args,args.len());

	if args.len() < 2 { eprintln!("Usage: {} filename <count>", args[0]); process::exit(1); }

  // Create a path to the desired file
    let path = Path::new(&args[1]);
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
		let mut tokens = line_data.split_whitespace();
		let vertex = tokens.next().unwrap().parse::<u32>().unwrap();
		let adjacent : Vec<u32> = tokens.map(|x| x.to_string().parse::<u32>().unwrap()).collect();
		

		let mut other : u32 = 0;
//		g.create_vertex(&vertex);
		for other_v in &adjacent {

			other = other_v.clone();
			let _num_edges = g.add_edge(vertex,*other_v);
		
		}
		if _count % 100000 == 0 {
			println!(" {} : {}  from {}  to {}" ,_count,line_data,vertex,other);
			io::stdout().flush().unwrap();
		}
		if _count % 10000 == 0 {
			print!(".");
			io::stdout().flush().unwrap();
		} 
    }
	let child = thread::Builder::new().stack_size(512 * 1024 * 1024).spawn(move || { 
	   // code to be executed in thread

		println!("Read {} lines",_count);
	//	g.print_vertexes();
	//	g.dfs_incoming(1,1,0);
	//	println!("Finish Order {:?}", g.finished_order);
	//	println!("Starting Vertex {:?}", g.start_search);
		g.finished_order = Vec::<u32>::new();
		g.start_search = HashMap::<u32,Vec::<u32>>::new();
		g.explored = HashMap::<u32,bool>::new();
		let list : Vec<u32> = g.vertex_map.keys().cloned().collect();
		g.dfs_loop_incoming(&list);
	//	println!("Finish Order {:?}", g.finished_order);
	//	println!("Starting Vertex {:?}", g.start_search);
		let list : Vec<u32> = g.finished_order.iter().rev().cloned().collect();
		g.dfs_loop_outgoing(&list);
		println!("\n Start search has {} entries",g.start_search.len());
		// println!("\n Start search {:?} entries",g.start_search);
		println!("\n Top Counts {:?} entries",g.top_search_cnts);
		let mut top_search_count_vec : Vec::<(u32, usize)> = g.top_search_cnts.iter().map(|(k,v)| (*k, *v)).collect();
		top_search_count_vec.sort_by(|a, b| b.1.cmp(&a.1));
		println!("\n Top Counts {:?} entries",top_search_count_vec);
	}).unwrap(); 
	child.join().unwrap();
//	println!("Starting Vertex {:?}", g.start_search);
}


/*
 * the rest of this file sets up unit tests
 * to run these, the command will be:
 * cargo test --package rust-template -- --nocapture
 * Note: 'rust-template' comes from Cargo.toml's 'name' key
 */

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
