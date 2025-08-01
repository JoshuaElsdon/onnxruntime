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

      if ((actual_input_name == "B" || actual_input_name == "zero_points") &&
          tensor_info.qnn_data_type == QNN_DATATYPE_UINT_8) {
        std::string casted_name = actual_input_name + "_casted_int32";
        LOGS(logger, INFO) << "Casting tensor " << actual_input_name << " to int32 as " << casted_name;

        ORT_RETURN_IF_ERROR(qnn_model_wrapper.CastStaticTensorToInt32(actual_input_name, casted_name, logger));
        actual_input_name = casted_name;
      }
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
                                                         const logging::Logger& logger,
                                                         bool do_op_validation) const {
  LOGS(logger, INFO) << "MatMulNBitsOpBuilder::ProcessAttributesAndOutputs() called.";

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  // Step 1: Identify output name
  const auto& output_def = node_unit.Outputs()[0];
  std::string input_to_quantize = output_def.node_arg.Name();  // typically "Y_QuantizeLinear_Input"
  std::string output_name = input_to_quantize;                 // default unless QuantizeLinear is found
  const Node* quantize_node = nullptr;

  // Step 2: Look for QuantizeLinear node that consumes this output
  for (const auto& node : graph_viewer.Nodes()) {
    if (node.OpType() == "QuantizeLinear" &&
        !node.InputDefs().empty() &&
        node.InputDefs()[0]->Name() == input_to_quantize) {
      quantize_node = &node;
      output_name = node.OutputDefs()[0]->Name();  // typically "Y_QuantizeLinear_Output"
      break;
    }
  }

  // Step 3: Extract quantization info from QuantizeLinear node
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_UFIXED_POINT_16;
  QnnQuantParamsWrapper op_output_quant_param;
  if (quantize_node) {
    NodeUnitIODef::QuantParam qp{/*scale=*/*quantize_node->InputDefs()[1]};
    if (quantize_node->InputDefs().size() > 2) {
      qp.zero_point = quantize_node->InputDefs()[2];
    }
    NodeUnitIODef io_def{*quantize_node->InputDefs()[0], qp};
    ORT_RETURN_IF_ERROR(op_output_quant_param.Init(qnn_model_wrapper, io_def));
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Missing QuantizeLinear consumer for output: ", input_to_quantize);
  }

  // Step 4: Get shape from original node output
  TensorInfo output_info{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output_def, output_info));
  std::vector<uint32_t> op_output_shape = output_info.shape;

  // Step 5: Register output tensor under name that DequantizeLinear expects
  QnnTensorWrapper output_tensor_wrapper(output_name,
                                         QNN_TENSOR_TYPE_NATIVE,
                                         qnn_data_type,
                                         std::move(op_output_quant_param),
                                         std::move(op_output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)),
                    "Failed to add output tensor.");

  // Step 6: Scalar parameters
  int32_t bits = 2, block_size = 64, K = 3072, N = 3072;
  std::vector<std::string> param_tensor_names;

  auto add_scalar_param = [&](const std::string& name, int32_t value) -> Status {
    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_INT_32;
    scalar_param.int32Value = value;
    QnnParamWrapper param(node_unit.Index(), node_unit.Name(), name, scalar_param);
    param_tensor_names.push_back(param.GetParamTensorName());

    if (!qnn_model_wrapper.AddParamWrapper(std::move(param))) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to add scalar param: ", name);
    }
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(add_scalar_param("bits", bits));
  ORT_RETURN_IF_ERROR(add_scalar_param("block_size", block_size));
  ORT_RETURN_IF_ERROR(add_scalar_param("K", K));
  ORT_RETURN_IF_ERROR(add_scalar_param("N", N));

  std::string name = utils::GetNodeName(node_unit);
  std::string package_name = GetQnnOpPackageName("MatMulNBits");
  std::string op_type = "MatMulNBits";

  // Step 7: Logging
  LOGS(logger, INFO) << "\n=== MatMulNBits CreateQnnNode Inputs ===";
  LOGS(logger, INFO) << "Node name       : " << name;
  LOGS(logger, INFO) << "Package name    : " << package_name;
  LOGS(logger, INFO) << "Op type         : " << op_type;
  LOGS(logger, INFO) << "Inputs          :";
  for (const auto& in : input_names) {
    LOGS(logger, INFO) << "  - " << in;
  }
  LOGS(logger, INFO) << "Outputs         :";
  LOGS(logger, INFO) << "  - " << output_name;
  LOGS(logger, INFO) << "Params          :";
  for (const auto& param : param_tensor_names) {
    LOGS(logger, INFO) << "  - " << param;
  }
  LOGS(logger, INFO) << "Do validation   : " << (do_op_validation ? "true" : "false");
  LOGS(logger, INFO) << "=========================================\n";

  // Step 8: Create the fused QNN node
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                        name,
                        package_name,
                        op_type,
                        std::move(input_names),
                        {output_name},
                        std::move(param_tensor_names),
                        do_op_validation),
                    "Failed to create QNN MatMulNBits node.");

  return Status::OK();
}
}  // namespace qnn
}  // namespace onnxruntime
