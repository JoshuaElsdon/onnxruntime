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

  // Mapping from ONNX node input name to actual quantized tensor to use
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& input_def = inputs[i];
    const std::string& input_name = input_def.node_arg.Name();

    std::string actual_input_name = input_name;

    // If this is a DequantizeLinear node output, try to trace back to the original quantized tensor
    const Node* producer = qnn_model_wrapper.GetGraphViewer().GetProducerNode(input_name);
    if (producer && producer->OpType() == "DequantizeLinear") {
      const auto& quant_input_defs = producer->InputDefs();
      if (!quant_input_defs.empty()) {
        // Use the input to DequantizeLinear (i.e., the quantized tensor)
        actual_input_name = quant_input_defs[0]->Name();
        LOGS(logger, INFO) << "Replacing input " << input_name << " with quantized tensor " << actual_input_name;
      }
    }

    // Add tensor if not already in QNN model
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(actual_input_name)) {
      TensorInfo tensor_info;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def, tensor_info));

      QnnTensorWrapper tensor_wrapper;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(tensor_info, actual_input_name, tensor_wrapper));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tensor_wrapper)),
                        "Failed to add tensor: " + actual_input_name);
    } else {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << actual_input_name;
    }

    input_names.emplace_back(actual_input_name);
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


  TensorInfo output_info{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
  std::vector<uint32_t> op_output_shape = output_info.shape;
  QnnQuantParamsWrapper op_output_quant_param = output_info.quant_param.Copy();
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_UFIXED_POINT_16;


  QnnTensorWrapper output_tensor_wrapper(output_name,
                                         QNN_TENSOR_TYPE_APP_READ,
                                         qnn_data_type,
                                         std::move(op_output_quant_param),
                                         std::move(op_output_shape));

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)),
                    "Failed to add output tensor.");

  int32_t bits = 2, block_size = 64, K = 3072, N = 3072;  // hard coded for now.

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
                        "MatMulNBits",                       // <== correct op name
                        std::move(input_names),
                        {output_name},
                        std::move(param_tensor_names),
                        do_op_validation),
                    "Failed to create QNN MatMulNBits node.");

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
