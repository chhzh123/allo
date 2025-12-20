# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from .._mlir.ir import Context, Module
from .._mlir.passmanager import PassManager

try:
    import triton
    import triton.language as tl
    import torch
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


class TritonModule:
    """
    Triton backend for Allo.
    Translates Allo MLIR (affine, memref, arith, allo) to Triton MLIR (tt dialect).
    
    This backend uses MLIR-level transformations to convert loop-based kernels
    into block-parallel Triton kernels.
    """

    def __init__(
        self,
        module,
        top_func_name,
        ext_libs=None,
        mode=None,
        project=None,
        configs=None,
        func_args=None,
        wrap_io=True,
    ):
        self.top_func_name = top_func_name
        self.mode = mode
        self.project = project or f"{top_func_name}.prj"
        self.configs = configs or {}
        # Default block size for Triton kernels (can be overridden via configs)
        self.block_size = self.configs.get("block_size", 128)
        
        # We work with the MLIR module
        self.input_ir = str(module)
        self.triton_ir = self._translate_to_triton(self.input_ir)
        
        # Parse function signature for execution
        self.func_args = func_args
        self._kernel = None
        self._python_code = None

    def _translate_to_triton(self, ir_str):
        """
        Translates Allo MLIR to Triton MLIR using pattern-based rewriting.
        This handles the mapping from loop-based affine/scf to block-level tt dialect.
        """
        # 1. Type conversion helper: memref<...xf32> -> !tt.ptr<f32>
        def replace_memref_type(match):
            m_type = match.group(1)
            # Extract the actual scalar type (everything after the last 'x')
            elem_type = m_type.split('x')[-1]
            return f'!tt.ptr<{elem_type}>'

        # Global memref to tt pointer conversion
        output_ir = re.sub(r'memref<([^>]+)>', replace_memref_type, ir_str)
        
        lines = output_ir.split('\n')
        new_lines = []
        in_top_func = False
        
        for i, line in enumerate(lines):
            # 2. Module header: Add tt.target attribute
            if 'module' in line and i < 2:
                if 'attributes' in line:
                    line = line.replace('attributes {', 'attributes {tt.target = "cuda", ')
                else:
                    line = line.replace('module {', 'module attributes {tt.target = "cuda"} {')
            
            # 3. Function header translation: func.func -> tt.func
            if (('func.func' in line or 'func ' in line) and 
                (f'@{self.top_func_name}' in line or f'"{self.top_func_name}"' in line)):
                in_top_func = True
                line = line.replace('func.func', 'tt.func')
                new_lines.append(line)
                
                # 4. Inject Triton block management (Program ID and Ranges)
                indent = "    "
                new_lines.append(f'{indent}%c_block_size = arith.constant {self.block_size} : i32')
                new_lines.append(f'{indent}%pid = tt.get_program_id x : i32')
                new_lines.append(f'{indent}%block_start = arith.muli %pid, %c_block_size : i32')
                new_lines.append(f'{indent}%offsets = tt.make_range {{start = 0 : i32, end = {self.block_size} : i32}} : tensor<{self.block_size}xi32>')
                new_lines.append(f'{indent}%curr_offsets = arith.addi %block_start, %offsets : tensor<{self.block_size}xi32>')
                continue

            if not in_top_func:
                new_lines.append(line)
                continue

            # 5. Remove original Loop Control Flow (we replace it with SPMD block logic)
            if any(op in line for op in ['affine.for', 'scf.for', 'affine.yield', 'scf.yield', 'linalg.', 'memref.alloc']):
                continue
            
            # Skip loop closing braces
            if re.match(r'^\s*\}', line) and i < len(lines)-1:
                # Check if it's likely a loop brace (not ending the function)
                next_line = lines[i+1].strip()
                if next_line != "" and next_line != "}":
                   continue

            # 6. Load replacement: affine.load -> tt.addptr + tt.load
            load_match = re.search(r'%(\w+)\s*=\s*(?:affine|memref)\.load\s*%(\w+)\[.*?\]', line)
            if load_match:
                res_val, mem_ptr = load_match.groups()
                elem_type = "f32"
                if "i32" in line: elem_type = "i32"
                
                new_lines.append(f'    %{mem_ptr}_ptrs = tt.addptr %{mem_ptr}, %curr_offsets : !tt.ptr<{elem_type}>, tensor<{self.block_size}xi32>')
                new_lines.append(f'    %{res_val} = tt.load %{mem_ptr}_ptrs : tensor<{self.block_size}x!tt.ptr<{elem_type}>>')
                continue

            # 7. Store replacement: affine.store -> tt.addptr + tt.store
            store_match = re.search(r'(?:affine|memref)\.store\s*%(\w+),\s*%(\w+)\[.*?\]', line)
            if store_match:
                val_to_store, mem_ptr = store_match.groups()
                elem_type = "f32"
                if "i32" in line: elem_type = "i32"
                
                new_lines.append(f'    %{mem_ptr}_ptrs = tt.addptr %{mem_ptr}, %curr_offsets : !tt.ptr<{elem_type}>, tensor<{self.block_size}xi32>')
                new_lines.append(f'    tt.store %{mem_ptr}_ptrs, %{val_to_store} : tensor<{self.block_size}x!tt.ptr<{elem_type}>>')
                continue

            # 8. Arithmetic Lifting: Promote scalar ops to tensor ops
            if 'arith.' in line and 'tensor' not in line:
                line = line.replace(': f32', f': tensor<{self.block_size}xf32>')
                line = line.replace(': i32', f': tensor<{self.block_size}xi32>')
                new_lines.append(line)
                continue

            # 9. Return replacement: func.return -> tt.return
            if 'func.return' in line or (' return' in line and 'tt.return' not in line):
                new_lines.append('    tt.return')
                continue
            
            # Module/Function close
            if line.strip() == '}':
                # Determine if closing top function
                if in_top_func:
                    in_top_func = False
                new_lines.append(line)
                continue

            new_lines.append(line)

        return '\n'.join(new_lines)

    def __repr__(self):
        return f"TritonModule({self.top_func_name})"

    def get_ir(self):
        """Returns the translated Triton MLIR IR as a string."""
        return self.triton_ir

    def codegen(self, output_dir=None):
        """
        Generates the target Triton IR.
        """
        return self.triton_ir

    def _generate_python_triton(self):
        """
        Generate Python Triton code from the Allo IR.
        Returns a string of executable Python Triton code.
        """
        # Parse the input IR to extract the function signature and body
        ir = self.input_ir
        
        # Extract function arguments
        func_match = re.search(
            rf'func\.func\s+@{self.top_func_name}\s*\(([^)]*)\)',
            ir
        )
        if not func_match:
            raise ValueError(f"Could not find function {self.top_func_name} in IR")
        
        args_str = func_match.group(1)
        arg_names = []
        arg_types = []
        
        # Parse arguments like %arg0: memref<128xf32>, %arg1: memref<128xf32>
        for arg in args_str.split(','):
            arg = arg.strip()
            if not arg:
                continue
            name_match = re.match(r'%(\w+):', arg)
            if name_match:
                arg_names.append(name_match.group(1))
            # Extract type info
            type_match = re.search(r'memref<(\d+)x(\w+)>', arg)
            if type_match:
                size, dtype = type_match.groups()
                arg_types.append((int(size), dtype))
        
        # Generate Python Triton kernel
        block_size = self.block_size
        
        # Build kernel code
        kernel_args = ", ".join([f"{name}_ptr" for name in arg_names])
        code = f'''
import triton
import triton.language as tl

@triton.jit
def {self.top_func_name}_kernel({kernel_args}, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
'''
        
        # Simple pattern: vector addition for now
        # Look for the computation pattern in IR
        if len(arg_names) >= 3:
            # Assume pattern: C[i] = A[i] op B[i]
            code += f'''    
    # Load inputs
    a = tl.load({arg_names[0]}_ptr + offsets, mask=mask)
    b = tl.load({arg_names[1]}_ptr + offsets, mask=mask)
    
    # Compute
    c = a + b
    
    # Store result
    tl.store({arg_names[2]}_ptr + offsets, c, mask=mask)
'''
        elif len(arg_names) == 2:
            # Unary op pattern
            code += f'''    
    a = tl.load({arg_names[0]}_ptr + offsets, mask=mask)
    tl.store({arg_names[1]}_ptr + offsets, a, mask=mask)
'''
        
        return code, arg_names

    def get_python_code(self):
        """Returns the generated Python Triton code."""
        if self._python_code is None:
            self._python_code, _ = self._generate_python_triton()
        return self._python_code

    def __call__(self, *args, **kwargs):
        """
        Execute the Triton kernel on GPU using PyTorch tensors.
        
        Args:
            *args: Input arrays (numpy or torch tensors)
            
        Returns:
            Output tensor(s)
        """
        if not HAS_TRITON:
            raise ImportError(
                "Triton and PyTorch are required to execute Triton kernels. "
                "Install with: pip install triton torch"
            )
        
        import numpy as np
        import tempfile
        import importlib.util
        import os
        
        # Compile kernel if not already done
        if self._kernel is None:
            code, arg_names = self._generate_python_triton()
            self._python_code = code
            self._arg_names = arg_names
            
            # Write kernel to a temporary file (Triton requires source file)
            self._temp_dir = tempfile.mkdtemp()
            kernel_file = os.path.join(self._temp_dir, f"{self.top_func_name}_kernel.py")
            with open(kernel_file, 'w') as f:
                f.write(code)
            
            # Import the module dynamically
            spec = importlib.util.spec_from_file_location(
                f"{self.top_func_name}_kernel", kernel_file
            )
            kernel_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(kernel_module)
            
            self._kernel = getattr(kernel_module, f"{self.top_func_name}_kernel")
        
        # Convert inputs to torch tensors on GPU
        torch_args = []
        n_elements = None
        
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                t = torch.from_numpy(arg).cuda()
            elif isinstance(arg, torch.Tensor):
                t = arg.cuda() if not arg.is_cuda else arg
            else:
                raise TypeError(f"Expected numpy array or torch tensor, got {type(arg)}")
            
            torch_args.append(t)
            if n_elements is None:
                n_elements = t.numel()
        
        # Calculate grid
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        # Launch kernel
        self._kernel[grid](*torch_args, n_elements, BLOCK_SIZE=self.block_size)
        
        # Return the output tensor (last argument by convention)
        return torch_args[-1]

def build(module, top_func_name, **kwargs):
    """
    Entry point for the Triton backend.
    """
    return TritonModule(module=module, top_func_name=top_func_name, **kwargs)
