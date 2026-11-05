use std::{
    fs,
    io::{
        self,
        BufRead
    },
    path::Path,
};

#[derive(Debug)]
pub enum ParseError {
    InvalidPath,
    IoFailure,
}

pub fn parse_csv(path: &str, delimeter: char
    ) -> Result<Vec<Vec<f64>>, ParseError> {
    let path = Path::new(path);
    let mut contents: Vec<Vec<f64>> = Vec::new();

    let file = fs::File::open(path)
        .map_err(|_err| ParseError::InvalidPath)?;

    let mut reader = io::BufReader::new(file);
    let mut buffer = String::new();

    loop {
        match reader.read_line(&mut buffer) {
            Ok(bytes) => {
                if bytes == 0 && buffer.len() == 0 {
                    break;
                }
                let mut row = Vec::<f64>::new();
                buffer
                    .split(delimeter)
                    .for_each(|item| {
                        match item.trim().parse::<f64>() {
                            Ok(x) => row.push(x),
                            Err(_) => {println!("{:?}", item)},
                        }
                    });
                buffer.clear();
                if row.len() != 0 {
                    contents.push(row);
                }
            }
            Err(_) => {
                return Err(ParseError::IoFailure);
            }
        }
    }

    Ok(contents)
}
