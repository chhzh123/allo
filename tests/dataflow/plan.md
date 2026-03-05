# Implementation Plan: Dato Linear Layout Synthesis via $\mathbb{F}_2$ Partitioning

**Objective:** Implement an automated compiler pipeline that maps high-level, unrolled Python array operations (e.g., FFT butterflies) onto physical FPGA memory banks without access conflicts, using $\mathbb{F}_2$ linear algebra.

## Phase 1: Frontend Mathematical Extraction (Code $\to \mathbb{F}_2$)

**Goal:** Lower the user's `@df.kernel` Python logic into an internal affine representation over the Boolean field $\mathbb{F}_2$.

* **1.1 AST Parsing and Symbolic Tracing:**
    * Intercept the Dato Python kernel to extract the loop iteration domain and memory access indices.
    * Identify loop induction variables $\vec{x}$ (e.g., $s$ for stage, $b$ for butterfly index).
* **1.2 Affine Linearization:**
    * Convert every array access into a formal affine transformation map: 
        $$\vec{y} = A\vec{x} \oplus \vec{c}$$
    * For operations containing bitwise XOR or stride patterns based on powers of two (common in FFTs), map the stride $2^s$ to the one-hot vector $\vec{e}_s$. 
    * *Implementation Note:* Store these mappings in a tabular format where columns represent the bit-positions of the iteration vector $\vec{x}$ and rows represent the access vector $\vec{y}$.

## Phase 2: Conflict Subspace Construction ($P$)

**Goal:** Formally define the "Conflict Subspace" ($P$) for each loop nest to identify which memory accesses happen concurrently in the hardware.

* **2.1 Extract Spatial Parallelism ($V_{space}$):**
    * Analyze the unroll factors applied to the kernel. If the compiler unrolls the inner loop by a factor of $K = 2^k$, identify which bits of the index are actively varying across these $K$ parallel instances.
    * Construct the span. For example, if 4 contiguous elements are processed, $V_{space} = \text{span}(\vec{e}_0, \vec{e}_1)$.
* **2.2 Extract Pattern Stride ($V_{pattern}$):**
    * Analyze the intra-cycle data requirements of the datapath logic (e.g., the two inputs to a butterfly PE). 
    * If a PE simultaneously requires index $\vec{y}$ and $\vec{y} \oplus \vec{e}_s$, the required simultaneous stride is $\vec{e}_s$. Set $V_{pattern} = \text{span}(\vec{e}_s)$.
* **2.3 Subspace Union:**
    * Compute the total conflict subspace for stage $s$:
        $$P_s = \text{span}(V_{space} \cup V_{pattern})$$
    * Store $P_s$ as a boolean matrix representing the basis vectors of the conflicting access patterns.

## Phase 3: Solving for the Partition Matrix ($S$)

**Goal:** Synthesize the Swizzle Matrix $S$ that maps a logical address to a physical memory bank ID such that no two simultaneous accesses map to the same bank.

* **3.1 Hardware Capacity Check:**
    * Given target hardware with $N_{banks} = 2^k$ available memory banks, check the dimensionality of the conflict subspace.
    * **Condition:** If $\dim(P_s) > k$, an unresolvable resource violation exists.
    * **Fallback:** Automatically trigger a compiler rewrite to reduce the unroll factor ($V_{space}$) until $\dim(P_s) \le k$.
* **3.2 Null Space Resolution (The Solver):**
    * The matrix $S$ must satisfy the condition that no non-zero vector in $P_s$ maps to the zero vector (which would indicate a bank collision):
        $$\forall \vec{v} \in P_s, \vec{v} \neq \vec{0} \implies S \vec{v} \neq \vec{0}$$
    * Solve for $S$ by finding a basis for the dual space of $P_s$.
* **3.3 Matrix Selection Heuristics:**
    * To minimize physical routing complexity in the generated RTL/HLS, prioritize selecting identity rows (direct wiring of LSBs) for $S$.
    * Only introduce XOR rows (e.g., $S = [\vec{e}_0, \vec{e}_1 \oplus \vec{e}_2]$) when necessary to resolve rank deficiencies in the null space intersection.

## Phase 4: Re-rolling and HLS Code Generation

**Goal:** Translate the optimized mathematical model back into synthesizable hardware code (e.g., C++ for HLS), embedding the physical bank partitioning.

* **4.1 Isomorphism Detection for Pipelining:**
    * Compare the dependency matrices $A_s$ across all stages $s = 0 \dots \log_2 N$.
    * Detect structurally isomorphic stages (where $A_{s+1}$ is a bit-shifted equivalent of $A_s$).
    * Instead of generating a fully unrolled spatial monolithic block, re-roll the isomorphic stages into an outer pipeline loop (`for s in range(LOG_N)`).
* **4.2 Swizzle Injection:**
    * Replace all logical memory accesses with the physically partitioned hardware addressing scheme using the solved matrix $S_s$.
    * Compute the Physical Bank ID: `bank_id = S_s @ idx` (implemented as hardware XOR trees).
    * Compute the Local Bank Offset: `offset = idx >> k`.
* **4.3 Emit Synthesizable Code:**
    * Generate the final hardware loop nests.
    * Apply appropriate backend pragmas (e.g., `#pragma HLS array_partition variable=banks complete dim=1` and `#pragma HLS pipeline II=1`) to ensure the spatial loop ($V_{space}$) executes fully in parallel within the bounds of the hardware clock cycle.