# Clock gating

This module provides automatic insertion of clock gates for reducing power usage.
It uses ABC for proving correctness of gating conditions.
When invoked, this module:
1. Gathers all registers in the design
2. For each register, it:
  a. Gathers all the connected nets in a BFS manner up to a certain limit (user-specified with a default of 100).
     The limit is there to make it faster, as gathering more nets empirically resulted in far worse execution time with no real result quality improvements.
     It also doesn't go across registers, i.e. if it encounters a register, it doesn't go through it, it stops and goes in different directions.
  b. Exports this gathered network to ABC.
  c. Checks if all the nets in the network can form a correct gating condition.
     It checks for a clock enable condition by ORing them, and a clock disable condition by ANDing them.
     The check is performed using ABC on the exported network:
     - First, simulation with random stimuli quickly checks for counterexamples.
     - Then, if no counterexample was found, a SAT solver is employed to prove that the gating condition is correct.
  d. If the set of all nets doesn't form a correct gating condition, it moves on to the next register.
     Otherwise, it:
     - Checks if after removing half of the nets, the gating condition still works.
     - If so, it drops the other half of the nets. Otherwise, it recurses into the other half of the nets to minimize that set.
     - Then, it recurses into the first half of the nets.
     This process produces a minimal set of nets that form a gating condition (not necessarily optimal).
  e. Adds the minimal set of nets with the corresponding gated register to a list of accepted gating conditions if it doesn't contain it already.
     If it does, it just adds the gated register to the list next to the pre-existing condition.
  f. Additionally, before checking a new gating condition (before b.), it checks if a previously accepted condition is suitable for the register.
     If so, it adds the register to the accepted list next to the pre-existing condition.
3. For each accepted gating condition, inserts a new clock gate that gates the corresponding registers under the gating condition.
   If the number of registers that can be gated by the condition is fewer than a certain user-configurable number (by default 10), it doesn't insert the clock gate and moves on to the next accepted condition.

Usage:

```tcl
read_liberty path/to/pdk/cell/library.lib
read_db path/to/your/design.odb
read_sdc path/to/your/constraints.sdc

clock_gating
```

## Commands

```{note}
All parameters for clock gating are optional, as indicated by square brackets: `[-param param]`.
```

### Clock gating

```tcl
clock_gating
    [-min_instances min_instances]
    [-max_cover max_cover]
    [-dump_dir dump_dir]
```

#### Options

| Switch Name | Description |
| ----- | ----- |
| `-min_instances` | Minimum number of instances that should be gated by a single clock gate. |
| `-max_cover` | Maximum number of initial gate condition candidate nets per instance. |
| `-dump_dir` | Name of the directory for debug dumps. |

## Example scripts

Example script on running `cgt` for a sample design of `aes` can be found here:

```
./test/aes_nangate45.tcl
```

## Regression tests

There are a set of regression tests in `./test`. For more information, refer to this [section](../../README.md#regression-tests).

Simply run the following script:

```shell
./test/regression
```

## Limitations

Clock gating is not supported for designs that contain cells not supported by ABC, such as adders.

## FAQs

Check out [GitHub discussion](https://github.com/The-OpenROAD-Project/OpenROAD/discussions/categories/q-a?discussions_q=category%3AQ%26A+clock-gating)
about this tool.

## References

1. Aaron P. Hurst. 2008. Automatic synthesis of clock gating logic with controlled netlist perturbation. In Proceedings of the 45th annual Design Automation Conference (DAC '08). Association for Computing Machinery, New York, NY, USA, 654–657. https://doi.org/10.1145/1391469.1391637

