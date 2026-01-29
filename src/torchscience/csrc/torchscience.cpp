// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>

extern "C" {
  PyObject* PyInit__csrc(void) {
    static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_csrc",
      nullptr,
      -1,
      nullptr,
    };

    return PyModule_Create(&module_def);
  }
}

// The main TORCH_LIBRARY declaration. Must exist exactly once.
// All schema definitions are in category-specific files using TORCH_LIBRARY_FRAGMENT.
TORCH_LIBRARY(torchscience, module) {}
