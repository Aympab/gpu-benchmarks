# How to build:

1. Build google benchmark apart:
   1. CMake command: `cmake .. -DBENCHMARK_ENABLE_TESTING=OFF -DCMAKE_BUILD_TYPE=Release`
2. Feed benchmark dir to `compile` script
   1. Example: `./compile --hw cpu --benchmark_BUILD=/path/tpls/benchmark/build`