# IZPR: Instant Zero Knowledge Proof of Reserve 

## Description
This is the source code of the IZPR Project,
presented at CoDecFin'24 (
T. Conley, N. Diaz, D. Espada, A. Kuruvilla, S. Mayne, and X. Fu.
"IZPR: Instant Zero Knowledge Proof of Reserve".
Pre-print at https://eprint.iacr.org/2023/1156.pdf).

Based on the recent progress in pre-processed lookup argument,
we present a system that can prove the asset of a financial
organization, instantly and incrementally, in zero knowledge.
The prover complexity is O(nlog(n)), where n is the
platform throughput (e.g., 7 for BTC and 2000 for Visa network).
The proof size and verifier cost are O(1).
Concretely, running with 32-threads on a 5 dollar/hr 
Google Cloud C2D instance, it supports throughput of
1024 transactions per second. The proof size is 3.4kb.


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
in the /dependency with some slight modifications for 
retrieving some of its private data members in structs
and resolving rust compiler incompatibility. If 
backporting, search "Modified by" and check the changes made.

## TODOs
* The zero knowledge cq protocl in this version uses a combination
of zk-VPD scheme (Y. Zhang, D. Genkin, J. Katz, D. Papadopoulos,
and C. Papamanthou. "A Zero-Knowlede Version of
vSQL", available at: https://eprint.iacr.org/2017/1146.pdf),
and $k$-bounded leaking KZG poly commits.
The proof size and prover speed can be improved about 1x
if the cq+ protocol (M. Campanelli, A. Fanoi, D. Fiore,
T. Li and H. Lipmaa, "Lookup Arguments: Improvements,
Extensions, and Applications to Zero-Knowledge Decision Trees".
available at: https://hal.science/hal-04234948/document) is used.

* We have another zk-CQ protocol using QA-NIZK with similar
cost to cq+. Source code to be released later.


## Authors
Trevor Conley, Nilsso Diaz, Diego Espada, Alvin Kuruvilla, Xiang
Fu (Xiang.Fu@hofstra.edu). 


## License

This project is licensed under the Apache and Open MIT License.

## Acknowledgments

* We implemented a zk-derivative of the CQ protocol by L. Eagen
D. Fiore, and A. Gabizon (https://eprint.iacr.org/2022/1763.pdf).
* The group FFT algorithm is adapted from arkworks' FFT algorithm
for field elements.
