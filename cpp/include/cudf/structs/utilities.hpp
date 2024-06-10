/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

/**
 * @file
 * @brief Class definition for cudf::struct_view.
 */

#pragma GCC visibility push(default)
namespace cudf::structs {

/**
 * @brief Superimpose nulls from a given null mask into the input column, using bitwise AND.
 *
 * This function will recurse through all struct descendants. It is expected that the size of
 * the given null mask in bits is the same as size of the input column.
 *
 * Any null strings/lists in the input (if any) will also be sanitized to make sure nulls in the
 * output always have their sizes equal to 0.
 *
 * @param null_mask Null mask to be applied to the input column
 * @param null_count Null count in the given null mask
 * @param input Column to apply the null mask to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate new device memory
 * @return A new column with potentially new null mask
 */
[[nodiscard]] std::unique_ptr<column> superimpose_nulls(bitmask_type const* null_mask,
                                                        size_type null_count,
                                                        std::unique_ptr<column>&& input,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr);

}  // namespace cudf::structs
#pragma GCC visibility pop
