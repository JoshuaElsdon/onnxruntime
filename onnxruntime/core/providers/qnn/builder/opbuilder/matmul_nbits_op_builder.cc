// File: matmul_nbits_op_builder.cc

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class MatMulNBitsOpBuilder : public BaseOpBuilder {
 public:
  MatMulNBitsOpBuilder() : BaseOpBuilder("MatMulNBitsOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulNBitsOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names, bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

void CreateMatMulNBitsOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<MatMulNBitsOpBuilder>());
}

///////////////////////////////////////////////////////////////////////////////
// Actual logic

Status MatMulNBitsOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                           const logging::Logger& logger,
                                           std::vector<std::string>& input_names,
                                           bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  LOGS(logger, INFO) << "MatMulNBitsOpBuilder::ProcessInputs() called.";
  const auto& inputs = node_unit.Inputs();

  for (const auto& input_def : inputs) {
    const std::string& input_name = input_def.node_arg.Name();
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
      QnnTensorWrapper tensor_wrapper;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, tensor_wrapper));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tensor_wrapper)), "Failed to add tensor: " + input_name);
    } else {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_name;
    }
    input_names.emplace_back(input_name);
  }

  return Status::OK();
}

Status MatMulNBitsOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                         const NodeUnit& node_unit,
                                                         std::vector<std::string>&& input_names,
                                                         const logging::Logger& /*logger*/,
                                                         bool do_op_validation) const {
  LOGS_DEFAULT(INFO) << "MatMulNBitsOpBuilder::ProcessAttributesAndOutputs() called.";
  const auto& outputs = node_unit.Outputs();
  const std::string& output_name = outputs[0].node_arg.Name();

  TensorInfo output_info;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info));

  // Convert shape to uint32_t
  std::vector<uint32_t> output_shape_uint32;
  output_shape_uint32.reserve(output_info.shape.size());
  for (auto dim : output_info.shape) {
    output_shape_uint32.push_back(static_cast<uint32_t>(dim));
  }

  QnnTensorWrapper output_tensor_wrapper(output_name,
                                         QNN_TENSOR_TYPE_APP_READ,  // graph output
                                         output_info.qnn_data_type,
                                         output_info.quant_param.Copy(),
                                         std::move(output_shape_uint32));

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)), "Failed to add output tensor.");

  int32_t bits = 2, block_size = 64, K = 3072, N = 3072; // hard coded for now.

  std::vector<std::string> param_tensor_names;

  // bits
  {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_INT_32;
    scalar_param.int32Value = bits;

    QnnParamWrapper bits_param(node_unit.Index(), node_unit.Name(), "bits", scalar_param);
    param_tensor_names.push_back(bits_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(bits_param));
  }

  // block_size
  {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_INT_32;
    scalar_param.int32Value = block_size;

    QnnParamWrapper block_size_param(node_unit.Index(), node_unit.Name(), "block_size", scalar_param);
    param_tensor_names.push_back(block_size_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(block_size_param));
  }

  // K
  {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_INT_32;
    scalar_param.int32Value = K;

    QnnParamWrapper K_param(node_unit.Index(), node_unit.Name(), "K", scalar_param);
    param_tensor_names.push_back(K_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(K_param));
  }

  // N
  {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_INT_32;
    scalar_param.int32Value = N;

    QnnParamWrapper N_param(node_unit.Index(), node_unit.Name(), "N", scalar_param);
    param_tensor_names.push_back(N_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(N_param));
  }
  // Now create the node
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                        utils::GetNodeName(node_unit),
                        GetQnnOpPackageName("MatMulNBits"),  // <== correct package name
                        "MatMulNBits",  // <== correct op name
                        std::move(input_names),
                        {output_name},
                        std::move(param_tensor_names),
                        do_op_validation),
                    "Failed to create QNN MatMulNBits node.");

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
