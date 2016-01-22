- Problem:			RockSample
- Size:				(4,4)
- Max # of trials:	100
- Time per move:	unlimited
- Belief update:	particle (500 particles)
- Search depth:		90
- Random seed:		42
- Eta:				0.95
- Platform:			VMware Player v4.0.2 (Windows 7 64bit) RHEL 6,
					Lenovo T530, Intel Core i7-3820QM @ 2.7Ghz

|Implementation				|Time			|Comments											|
----------------------------|--------------:|---------------------------------------------------|
|C++ 						|0.78s			|													|
|Julia (without POMCP.jl)	|2.58s			|													|
|Julia (with POMCP.jl)		|5.69s			|Improved type specifications and var preallocation	|
