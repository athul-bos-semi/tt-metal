// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/concat_device_operation.hpp"

namespace ttnn::operations::test_ops {

    struct ExecuteConcat {
        // static Tensor execute_on_worker_thread (const std::vector<Tensor> &input_tensors, uint8_t dim) {
        //     uint8_t queue_id = 0;
        //     return ttnn::device_operation::run<Concat>(
        //         queue_id,
        //         Concat::operation_attributes_t{dim},
        //         Concat::tensor_args_t{input_tensors}
        //     );
        // }

        static Tensor operator()(uint8_t queue_id, const std::vector<ttnn::Tensor> &input_tensors, const uint32_t dim) {
            return ttnn::device_operation::run<Concat>(
                queue_id,
                Concat::operation_attributes_t{.dim = dim},
                Concat::tensor_args_t{.input_tensors = input_tensors});
        }

        static Tensor operator()(const std::vector<ttnn::Tensor> &input_tensors, const uint32_t dim) { 
            return operator()(0, input_tensors, dim); 
        }
    };

}

namespace ttnn::operations::test_ops::test_concat {
    // constexpr auto testing_concat = ttnn::register_operation<ttnn::operations::test_ops::test_concat::ExecuteConcat>("ttnn::operations::test_ops::test_concat::testing_concat");
    constexpr auto testing_concat = ttnn::register_operation<"ttnn::operations::test_ops::test_concat::testing_concat",ttnn::operations::test_ops::ExecuteConcat>();
}
