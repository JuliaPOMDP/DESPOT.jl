- Max # of trials:	100
- Time per move:	unlimited
- Belief update:	particle (500 particles)
- Search depth:		90
- Random seed:		42
- Eta:				0.95
- Platform:			VMware Player v4.0.2 (Windows 7 64bit) RHEL 6,
					Lenovo T530, Intel Core i7-3820QM @ 2.7Ghz

The results below are averages of 5 runs per solver:

*RockSample, grid_size = 4, num_rocks = 4*

|Implementation				|Time			|Comments											|
----------------------------|--------------:|---------------------------------------------------|
|C++ 						|0.78s			|													|
|Julia (without POMDPs.jl)	|1.98s			|													|
|Julia (with POMDPs.jl)		|2.45s			|													|

*RockSample, grid_size = 7, num_rocks = 8*

|Implementation				|Time			|Comments											|
----------------------------|--------------:|---------------------------------------------------|
|C++ 						|7.46s			|													|
|Julia (without POMDPs.jl)	|27.54s			|				|
|Julia (with POMDPs.jl)		|33.58s			|				|

*RockSample, grid_size = 11, num_rocks = 11*

|Implementation				|Time			|Comments											|
----------------------------|--------------:|---------------------------------------------------|
|C++ 						|40.28s			|													|
|Julia (without POMDPs.jl)	|171.04s		|													|
|Julia (with POMDPs.jl)		|186.72s		|													|
