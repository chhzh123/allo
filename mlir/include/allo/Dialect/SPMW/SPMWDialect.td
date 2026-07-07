/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SPMW_DIALECT
#define SPMW_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// spmw dialect definition.
//===----------------------------------------------------------------------===//

def SPMW_Dialect : Dialect {
    let name = "spmw";
    let summary = "Single-program multiple-work-unit spatial dialect.";
    let description = [{
        The spmw dialect carries the regularity of a spatial accelerator as
        structure: a work unit replicated over a grid, wired by a topology, and
        specialized at the boundary by roles. Its central op, `spmw.map`, is a
        rolled description that survives to code generation instead of being
        expanded to one function per grid point.
    }];
    let cppNamespace = "::mlir::spmw";
}

//===----------------------------------------------------------------------===//
// Base spmw operation definition.
//===----------------------------------------------------------------------===//

class SPMW_Op<string mnemonic, list<Trait> traits = []> :
        Op<SPMW_Dialect, mnemonic, traits>;

#endif // SPMW_DIALECT
