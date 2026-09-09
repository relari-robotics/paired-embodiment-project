# YCB object 029: plate

`nontextured.stl` is the 16k-face Google scanner mesh of object **029_plate**
from the [YCB Object and Model Set](https://www.ycbbenchmarks.com/), downloaded
unmodified from
`https://ycb-benchmarks.s3.amazonaws.com/data/google/029_plate_google_16k.tgz`.
It is a watertight shell scan, about 26 cm across and 2.7 cm tall, in metres.

The scenario does not simulate this shell directly: a thin scanned wall lets a
pressed soft body tunnel through it. `scenario.py` converts it at load time
into a solid body (`solidify_plate`) by sampling the top surface on a 3 mm grid
and extruding it down to a flat base, keeping the exact dish profile and rim.

The YCB models are distributed by the YCB benchmark project for research use;
see the YCB website for their terms. The mesh is redistributed here only as a
scenario input and is not covered by this repository's Apache-2.0 licence.
