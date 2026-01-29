// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// CPU backend
#include "cpu/encryption/chacha20.h"
#include "cpu/encryption/sha256.h"
#include "cpu/encryption/sha3.h"
#include "cpu/encryption/blake2.h"
#include "cpu/encryption/aes.h"
#include "cpu/encryption/poly1305.h"
#include "cpu/encryption/chacha20_poly1305.h"
#include "cpu/encryption/curve25519.h"
#include "cpu/encryption/ed25519.h"
#include "cpu/encryption/pbkdf2.h"
#include "cpu/encryption/hkdf.h"
#include "cpu/encryption/shamir.h"
#include "cpu/encryption/additive.h"

// Meta backend
#include "meta/encryption/chacha20.h"
#include "meta/encryption/sha256.h"
#include "meta/encryption/sha3.h"
#include "meta/encryption/blake2.h"
#include "meta/encryption/aes.h"
#include "meta/encryption/poly1305.h"
#include "meta/encryption/chacha20_poly1305.h"
#include "meta/encryption/curve25519.h"
#include "meta/encryption/ed25519.h"
#include "meta/encryption/pbkdf2.h"
#include "meta/encryption/hkdf.h"
#include "meta/encryption/shamir.h"
#include "meta/encryption/additive.h"

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Stream ciphers
  m.def("chacha20(Tensor key, Tensor nonce, int num_bytes, int counter=0) -> Tensor");

  // Hash functions
  m.def("sha256(Tensor data) -> Tensor");
  m.def("sha3_256(Tensor data) -> Tensor");
  m.def("sha3_512(Tensor data) -> Tensor");
  m.def("keccak256(Tensor data) -> Tensor");
  m.def("blake2b(Tensor data, Tensor key, int digest_size=64) -> Tensor");
  m.def("blake2s(Tensor data, Tensor key, int digest_size=32) -> Tensor");

  // Block ciphers
  m.def("aes_encrypt_block(Tensor plaintext, Tensor key) -> Tensor");
  m.def("aes_decrypt_block(Tensor ciphertext, Tensor key) -> Tensor");
  m.def("aes_ctr(Tensor data, Tensor key, Tensor nonce, int counter=0) -> Tensor");

  // Message authentication codes
  m.def("poly1305(Tensor data, Tensor key) -> Tensor");

  // Authenticated encryption
  m.def("chacha20_poly1305_encrypt(Tensor plaintext, Tensor key, Tensor nonce, Tensor aad) -> (Tensor, Tensor)");
  m.def("chacha20_poly1305_decrypt(Tensor ciphertext, Tensor key, Tensor nonce, Tensor aad, Tensor tag) -> Tensor");

  // X25519 key exchange
  m.def("x25519(Tensor scalar, Tensor point) -> Tensor");
  m.def("x25519_base(Tensor scalar) -> Tensor");
  m.def("x25519_keypair(Tensor seed) -> (Tensor, Tensor)");

  // Ed25519 signatures
  m.def("ed25519_keypair(Tensor seed) -> (Tensor, Tensor)");
  m.def("ed25519_sign(Tensor private_key, Tensor message) -> Tensor");
  m.def("ed25519_verify(Tensor public_key, Tensor message, Tensor signature) -> Tensor");

  // Key derivation functions
  m.def("pbkdf2_sha256(Tensor password, Tensor salt, int iterations, int output_len) -> Tensor");
  m.def("hkdf_extract_sha256(Tensor salt, Tensor ikm) -> Tensor");
  m.def("hkdf_expand_sha256(Tensor prk, Tensor info, int output_len) -> Tensor");
  m.def("hkdf_sha256(Tensor ikm, Tensor salt, Tensor info, int output_len) -> Tensor");

  // Secret sharing
  m.def("shamir_split(Tensor secret, Tensor randomness, int n, int k) -> Tensor");
  m.def("shamir_reconstruct(Tensor shares, Tensor indices) -> Tensor");
  m.def("additive_split(Tensor secret, Tensor randomness, int n) -> Tensor");
  m.def("additive_reconstruct(Tensor shares) -> Tensor");
}
