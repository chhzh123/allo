template <typename T1, typename T2> inline T1 min(T1 x, T2 y) { return (x < T1(y)) ? x : T1(y); }
template <typename T1, typename T2> inline T1 max(T1 x, T2 y) { return (x > T1(y)) ? x : T1(y); }

#include <tapa.h>
using tapa::aligned_allocator;/* Helper Function */
void host_serialize_A(std::vector<char, aligned_allocator<char>> &A_to, std::vector<char, aligned_allocator<char>> &A_from){
  /* Variable Declaration */
  unsigned int cnt = 0;
  /* Variable Declaration */

  // array
  // io_L3
  for (int c3 = 0; c3 <= 31; c3 += 1) {
    // io_L2
    for (int c5 = 0; c5 <= 31; c5 += 1)
      A_to[cnt++] = A_from[c3 * 32 + c5];
  }
}
/* Helper Function */

/* Helper Function */
void host_serialize_B(std::vector<char, aligned_allocator<char>> &B_to, std::vector<char, aligned_allocator<char>> &B_from){
  /* Variable Declaration */
  unsigned int cnt = 0;
  /* Variable Declaration */

  // array
  // io_L3
  for (int c3 = 0; c3 <= 31; c3 += 1) {
    // io_L2
    for (int c5 = 0; c5 <= 31; c5 += 1)
      B_to[cnt++] = B_from[c3 * 32 + c5];
  }
}
/* Helper Function */

/* Helper Function */
void host_deserialize_C(std::vector<int, aligned_allocator<int>> &C_to, std::vector<int, aligned_allocator<int>> &C_from){
  /* Variable Declaration */
  unsigned int cnt = 0;
  /* Variable Declaration */

  // array
  // io_L3
  for (int c3 = 0; c3 <= 31; c3 += 1) {
    // io_L2
    for (int c4 = 0; c4 <= 31; c4 += 1) {
      // io_L1
      // pe
      C_to[c4 * 32 + c3] = C_from[cnt++];
    }
  }
}
/* Helper Function */

