#pragma once
// "Do not blow away" helpers: force the compiler to treat a value as
// observable, so it can't hoist/eliminate the loop we're trying to time.
// Same trick Chandler Carruth popularized for Google Benchmark internals.

namespace roofline {

template <typename T>
inline void do_not_optimize(T const& value) {
    asm volatile("" : : "g"(value) : "memory");
}

inline void clobber_memory() {
    asm volatile("" : : : "memory");
}

} // namespace roofline
