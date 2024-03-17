# IZPR: Instant Zero Knowledge Proof of Reserve 

## Description
This is the source code of the IZPR Project,
presented at CoDecFin'24 (https://eprint.iacr.org/2023/1156.pdf).


## Getting Started

### Dependencies

* Cargo/Rustc 1.7.3
* arkswork library (https://github.com/arkworks-rs/).

### Installing

* Run ./scripts/compile.sh

### Executing program

* Run python3 scripts/paperdata.py (or cd data; ./run.sh)


## Help
One specific version of the arkswork library is placed
in dependency with some slight modifications for 
retrieving some of its private data members in structs
and resolving rust compiler incompatibility. If 
backporting, search "Modified by" and check the changes made.

## TODOs
* The zero knowledge cq protocl in this version uses a combination
of zk-VPD scheme and $k$-bounded leaking KZG poly commits.
The proof size and prover speed can be improved about 1X
if the CQ+ protocol (https://hal.science/hal-04234948/document) is used.



## Authors

Contributors names and contact info
Trevor Conley, Nilsso Diaz, Diego Espada, Alvin Kuruvilla, Xiang
Fu (Xiang.Fu@hofstra.edu)


## License

This project is licensed under the Apache and Open MIT License - see the LICENSE.md file for details

## Acknowledgments

* We implemented a zk-derivative of CQ protocol by L. Eagen
D. Fiore, and A. Gabizon (https://eprint.iacr.org/2022/1763.pdf).
* The group FFT algorithm is adapted from arkworks' FFT algorithm
for field elements.
