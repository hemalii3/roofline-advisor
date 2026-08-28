# Deriving kernel arithmetic-intensity profiles

This is the trickiest and most interview-worthy part of the project: there's
no single "correct" way to count FLOPs/bytes for algorithms like `find` that
have data-dependent control flow. Document your convention clearly rather
than chasing a single "true" number -- reviewers care more that the
convention is principled and stated than that it's unimpeachable.

For each of the five algorithms, work out:

- **flops_per_elem**: floating-point (or integer, if that's more relevant to
  the kernel -- note the distinction in your writeup) ops per input element,
  in the *average* / expected case you're modeling.
- **bytes_per_elem**: bytes read + written per element, *on the device*,
  during the compute phase (not counting host<->device transfer).
- **transfer_bytes_per_elem**: bytes that must move across PCIe per input
  element, amortized across the whole call (input in, output out).

## Suggested conventions per algorithm

**for_each**: straightforward -- whatever the per-element functor does.
Use a simple example functor (e.g. `x = a*x + b`) consistent with what you
benchmarked in P1: 1 read + 1 write = 16 bytes (double), 2 flops (1 mul, 1
add). Transfer: input read once, output written once = 16 bytes -- unless
your P1 benchmark does the operation in-place, in which case transfer is
8 bytes in + 8 bytes out anyway since the whole buffer still crosses PCIe
both ways.

**reduce**: n elements read, O(1) written (a scalar). bytes_per_elem on
device ~= 8 (just the read, amortize the final write over n). Transfer:
n elements in, but only a scalar out -- transfer_bytes_per_elem ~=
8 + (8/n), which -> 8 for large n. flops_per_elem = 1 (one add per
combine step, ignoring the log-depth tree reduction's total work being the
same as a linear scan in a work-efficient implementation).

**find**: THE hard one, because of early termination. Two honest options:
  1. **Worst case**: model as if it scans everything (flops=0, since it's a
     comparison not a FLOP -- consider switching this kernel's "intensity"
     axis to ops/byte using integer compares rather than FLOP/byte, and
     say so explicitly).
  2. **Expected case with target found at position p**: bytes_per_elem
     effectively becomes n/p amplified -- i.e. intensity is a function of
     *where* the match is, not just n. This is worth a paragraph in your
     writeup: it's a case where "arithmetic intensity" as a single scalar
     per algorithm breaks down, which is itself a finding.
  Pick worst-case for the model's crossover prediction (conservative), but
  discuss case 2 in the report as a limitation.

**sort**: use the flops/bytes implied by your actual implementation
(radix vs comparison-based changes this a lot -- state which one P1 used).
For a comparison sort, O(n log n) comparisons is a reasonable basis;
convert "comparisons" to an effective flops-equivalent count consistent
with however you're already normalizing find's compares, so the two are
comparable on the same plot.

**inc_scan**: similar to for_each but with a sequential dependency; bytes
similar to reduce+write (read once, write once = 16 bytes), flops ~1 per
element (running add).

## Cross-checking

Once you have flops_per_elem/bytes_per_elem per algorithm, compute
`arithmetic_intensity = flops/bytes` and sanity check it against your P1
finding that **sort has the highest arithmetic intensity of the five and is
the only one that crosses over** -- if your derived numbers don't reproduce
that ordering, the accounting convention needs revisiting before trusting
the crossover predictions.
