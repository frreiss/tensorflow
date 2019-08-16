/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/util/byte_swap.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

template <typename T>
Tensor Constant(T v, TensorShape shape) {
  Tensor ret(DataTypeToEnum<T>::value, shape);
  ret.flat<T>().setConstant(v);
  return ret;
}

template <typename T>
Tensor Constant_2x3(T v) {
  return Constant(v, TensorShape({2, 3}));
}

// Type-specific subroutine of SwapBytes test below
template <typename T>
void TestByteSwap(const T* forward, const T* swapped, int array_len) {
  auto bytes_per_elem = sizeof(T);

  // Convert the entire array at once
  std::unique_ptr<T[]> forward_copy(new T[array_len]);
  std::memcpy(forward_copy.get(), forward, array_len * bytes_per_elem);
  TF_EXPECT_OK(ByteSwapArray(reinterpret_cast<char*>(forward_copy.get()),
                             bytes_per_elem, array_len));
  for (int i = 0; i < array_len; i++) {
    EXPECT_EQ(forward_copy.get()[i], swapped[i]);
  }

  // Then the array wrapped in a tensor
  auto shape = TensorShape({array_len});
  auto dtype = DataTypeToEnum<T>::value;
  Tensor forward_tensor(dtype, shape);
  Tensor swapped_tensor(dtype, shape);
  std::memcpy(const_cast<char*>(forward_tensor.tensor_data().data()), forward,
              array_len * bytes_per_elem);
  std::memcpy(const_cast<char*>(swapped_tensor.tensor_data().data()), swapped,
              array_len * bytes_per_elem);
  TF_EXPECT_OK(ByteSwapTensor(&forward_tensor));
  test::ExpectTensorEqual<T>(forward_tensor, swapped_tensor);
}

// A battery of tests of the routines in byte_swap.cc
TEST(ByteSwapTest, SwapBytes) {
  // Test patterns, manually swapped so that we aren't relying on the
  // correctness of our own byte-swapping macros when testing those macros.
  // At least one of the entries in each list has the sign bit set when
  // interpreted as a signed int.
  const int arr_len_16 = 4;
  const uint16_t forward_16[] = {0x1de5, 0xd017, 0xf1ea, 0xc0a1};
  const uint16_t swapped_16[] = {0xe51d, 0x17d0, 0xeaf1, 0xa1c0};
  const int arr_len_32 = 2;
  const uint32_t forward_32[] = {0x0ddba115, 0xf01dab1e};
  const uint32_t swapped_32[] = {0x15a1db0d, 0x1eab1df0};
  const int arr_len_64 = 2;
  const uint64_t forward_64[] = {0xf005ba11caba1000, 0x5ca1ab1ecab005e5};
  const uint64_t swapped_64[] = {0x0010baca11ba05f0, 0xe505b0ca1eaba15c};

  // 16-bit types
  TestByteSwap(forward_16, swapped_16, arr_len_16);
  TestByteSwap(reinterpret_cast<const int16_t*>(forward_16),
               reinterpret_cast<const int16_t*>(swapped_16), arr_len_16);
  TestByteSwap(reinterpret_cast<const bfloat16*>(forward_16),
               reinterpret_cast<const bfloat16*>(swapped_16), arr_len_16);

  // 32-bit types
  TestByteSwap(forward_32, swapped_32, arr_len_32);
  TestByteSwap(reinterpret_cast<const int32_t*>(forward_32),
               reinterpret_cast<const int32_t*>(swapped_32), arr_len_32);
  TestByteSwap(reinterpret_cast<const float*>(forward_32),
               reinterpret_cast<const float*>(swapped_32), arr_len_32);

  // 64-bit types
  // Cast to uint64*/int64* to make DataTypeToEnum<T> happy
  TestByteSwap(reinterpret_cast<const uint64*>(forward_64),
               reinterpret_cast<const uint64*>(swapped_64), arr_len_64);
  TestByteSwap(reinterpret_cast<const int64*>(forward_64),
               reinterpret_cast<const int64*>(swapped_64), arr_len_64);
  TestByteSwap(reinterpret_cast<const double*>(forward_64),
               reinterpret_cast<const double*>(swapped_64), arr_len_64);

  // Complex types.
  // Logic for complex number handling is only in ByteSwapTensor, so don't test
  // ByteSwapArray
  const float* forward_float = reinterpret_cast<const float*>(forward_32);
  const float* swapped_float = reinterpret_cast<const float*>(swapped_32);
  const double* forward_double = reinterpret_cast<const double*>(forward_64);
  const double* swapped_double = reinterpret_cast<const double*>(swapped_64);
  Tensor forward_complex64 = Constant_2x3<complex64>(
      std::complex<float>(forward_float[0], forward_float[1]));
  Tensor swapped_complex64 = Constant_2x3<complex64>(
      std::complex<float>(swapped_float[0], swapped_float[1]));
  Tensor forward_complex128 = Constant_2x3<complex128>(
      std::complex<double>(forward_double[0], forward_double[1]));
  Tensor swapped_complex128 = Constant_2x3<complex128>(
      std::complex<double>(swapped_double[0], swapped_double[1]));

  TF_EXPECT_OK(ByteSwapTensor(&forward_complex64));
  test::ExpectTensorEqual<complex64>(forward_complex64, swapped_complex64);

  TF_EXPECT_OK(ByteSwapTensor(&forward_complex128));
  test::ExpectTensorEqual<complex128>(forward_complex128, swapped_complex128);
}

}  // namespace

}  // namespace tensorflow
