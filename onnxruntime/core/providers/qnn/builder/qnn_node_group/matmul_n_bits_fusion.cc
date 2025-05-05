#include "core/providers/qnn/builder/qnn_node_group/matmul_n_bits_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>
#include <utility>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"

namespace onnxruntime {
namespace qnn {

// Forward declarations.
#define ValidateOnQnn(qnn_model_wrapper, input_dq_unit, matmul_n_bits_unit, output_q_unit, scale_dq_unit, logger) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (input_dq_unit), (matmul_n_bits_unit), (output_q_unit), (scale_dq_unit), true, logger)
#define CreateOnQnn(qnn_model_wrapper, input_dq_unit, matmul_n_bits_unit, output_q_unit, scale_dq_unit, logger) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (input_dq_unit), (matmul_n_bits_unit), (output_q_unit), (scale_dq_unit), false, logger)
static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& input_dq_unit, const NodeUnit& matmul_n_bits_unit, const NodeUnit& output_q_unit, const NodeUnit& scale_dq_unit, bool validate, const logging::Logger& logger);

std::unique_ptr<IQnnNodeGroup> MatMulNBitsQDQFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& input_dq_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  LOGS(logger, INFO) << "MatMulNBitsQDQFusion::TryFusion() called.";

  // Looking for a standalone Dequantize to start the sequence.
  if (input_dq_unit.OpType() != "DequantizeLinear" ||
      input_dq_unit.UnitType() != NodeUnit::Type::SingleNode) {
    LOGS(logger, INFO) << "MatMulNBitsQDQFusion::TryFusion() input_dq_unit is not a standalone DequantizeLinear node.";
    return nullptr;
  } else {
    LOGS(logger, INFO) << "MatMulNBitsQDQFusion::TryFusion() input_dq_unit is a standalone DequantizeLinear node.";
  }

  // Dequantize must have a single MatMulNBits child (1 output edge) and must not produce a graph output.
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  const std::array<std::string_view, 1> child_types = {"MatMulNBits"};
  const NodeUnit* matmul_n_bits_node_unit = GetOnlyChildOfType(graph_viewer, input_dq_unit, child_types,
                                                               node_to_node_unit, node_unit_to_qnn_node_group);

  if (matmul_n_bits_node_unit == nullptr) {
    LOGS(logger, INFO) << "MatMulNBitsQDQFusion::TryFusion() matmul_n_bits_node_unit is nullptr.";
    return nullptr;
  } else {
    LOGS(logger, INFO) << "MatMulNBitsQDQFusion::TryFusion() matmul_n_bits_node_unit is not nullptr.";
  }

  // the matmul_n_bits must have a Dequanize as input to its scale input (input index 2).
  const std::array<std::string_view, 1> input_types = {"DequantizeLinear"};
  const NodeUnit* scale_dq_node_unit = GetInputTypeOnIndex(graph_viewer, *matmul_n_bits_node_unit, 2, input_types, node_to_node_unit, node_unit_to_qnn_node_group, logger);

  if (scale_dq_node_unit == nullptr) {
    LOGS(logger, INFO) << "MatMulNBitsQDQFusion::TryFusion() scale_dq_node_unit is nullptr.";
    return nullptr;
  } else {
    LOGS(logger, INFO) << "MatMulNBitsQDQFusion::TryFusion() scale_dq_node_unit is not nullptr.";
  }

  // the matmul_nbits must have a single QuantizeLinear as output (1 output edge) and must not produce a graph output.
  const std::array<std::string_view, 1> output_types = {"QuantizeLinear"};
  const NodeUnit* output_q_node_unit = GetOnlyChildOfType(graph_viewer, *matmul_n_bits_node_unit, output_types,
                                                          node_to_node_unit, node_unit_to_qnn_node_group);

  if (output_q_node_unit == nullptr) {
    LOGS(logger, INFO) << "MatMulNBitsQDQFusion::TryFusion() output_q_node_unit is nullptr.";
    return nullptr;
  } else {
    LOGS(logger, INFO) << "MatMulNBitsQDQFusion::TryFusion() output_q_node_unit is not nullptr.";
  }

  if (Status status = ValidateOnQnn(qnn_model_wrapper, input_dq_unit, *matmul_n_bits_node_unit, *output_q_node_unit, *scale_dq_node_unit, logger);
      !status.IsOK()) {
    LOGS(logger, INFO) << "MatMulNBitsQDQFusion::TryFusion() NEARLY THERE:::validation failed.";
    return nullptr;
  } else {
    LOGS(logger, INFO) << "MatMulNBitsQDQFusion::TryFusion() NEARLY THERE:::validation succeeded.";
  }

  return std::make_unique<MatMulNBitsQDQFusion>(input_dq_unit, *matmul_n_bits_node_unit, *output_q_node_unit, *scale_dq_node_unit);
}

MatMulNBitsQDQFusion::MatMulNBitsQDQFusion(const NodeUnit& input_dq_unit, const NodeUnit& matmul_n_bits_unit, const NodeUnit& output_q_unit, const NodeUnit& scale_dq_unit)
    : node_units_{&input_dq_unit, &matmul_n_bits_unit, &output_q_unit, &scale_dq_unit} {
}

Status MatMulNBitsQDQFusion::IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  return ValidateOnQnn(qmw, *node_units_[0], *node_units_[1], *node_units_[2], *node_units_[3], logger);
}

Status MatMulNBitsQDQFusion::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  return CreateOnQnn(qmw, *node_units_[0], *node_units_[1], *node_units_[2], *node_units_[3], logger);
}

gsl::span<const NodeUnit* const> MatMulNBitsQDQFusion::GetNodeUnits() const {
  return node_units_;
}

const NodeUnit* MatMulNBitsQDQFusion::GetTargetNodeUnit() const {
  return node_units_[0];
}

static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& input_dq_unit,
                                    const NodeUnit& matmul_n_bits_unit,
                                    const NodeUnit& output_q_unit,
                                    const NodeUnit& scale_dq_unit,
                                    bool validate,
                                    const logging::Logger& logger) {
  LOGS(logger, INFO) << "CreateOrValidateOnQnn called. validate: " << validate;
  assert(matmul_n_bits_unit.OpType() == "MatMulNBits" && input_dq_unit.OpType() == "DequantizeLinear" &&
         output_q_unit.OpType() == "QuantizeLinear" && scale_dq_unit.OpType() == "DequantizeLinear");
  const auto& node_name = utils::GetNodeName(input_dq_unit);
  const NodeUnitIODef& a_input_def = input_dq_unit.Inputs()[0];
  const NodeUnitIODef& b_input_def = matmul_n_bits_unit.Inputs()[1];
  const NodeUnitIODef& scale_input_def = scale_dq_unit.Inputs()[0];
  const NodeUnitIODef& zeros_input_def = matmul_n_bits_unit.Inputs()[3];
  const NodeUnitIODef& output_def = output_q_unit.Outputs()[0];

  LOGS(logger, INFO) << " node_name: " << node_name;
  LOGS(logger, INFO) << " a_input_def: " << a_input_def.node_arg.Name();
  LOGS(logger, INFO) << " b_input_def: " << b_input_def.node_arg.Name();
  LOGS(logger, INFO) << " scale_input_def: " << scale_input_def.node_arg.Name();
  LOGS(logger, INFO) << " zeros_input_def: " << zeros_input_def.node_arg.Name();
  LOGS(logger, INFO) << " output_def: " << output_def.node_arg.Name();
  LOGS(logger, INFO) << " validate: " << validate;

  QnnTensorWrapper a_input_tensor, b_input_tensor, scale_input_tensor, zeros_input_tensor;
  QnnTensorWrapper output_tensor;

  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(a_input_def, a_input_tensor));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(b_input_def, b_input_tensor));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(scale_input_def, scale_input_tensor));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(zeros_input_def, zeros_input_tensor));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

  // currently there is only one valid set of paramters for this op.
  Qnn_Scalar_t bits_scalar;
  bits_scalar.dataType = QNN_DATATYPE_INT_32;
  bits_scalar.uint32Value = 2;
  Qnn_Scalar_t block_size_scalar;
  block_size_scalar.dataType = QNN_DATATYPE_INT_32;
  block_size_scalar.uint32Value = 64;
  Qnn_Scalar_t K_scalar;
  K_scalar.dataType = QNN_DATATYPE_INT_32;
  K_scalar.uint32Value = 3072;
  Qnn_Scalar_t N_scalar;
  N_scalar.dataType = QNN_DATATYPE_INT_32;
  N_scalar.uint32Value = 3072;
  QnnParamWrapper bits_wrapper(input_dq_unit.Index(), node_name, "bits", bits_scalar);
  QnnParamWrapper block_size_wrapper(input_dq_unit.Index(), node_name, "block_size", block_size_scalar);
  QnnParamWrapper K_wrapper(input_dq_unit.Index(), node_name, "K", K_scalar);
  QnnParamWrapper N_wrapper(input_dq_unit.Index(), node_name, "N", N_scalar);
  std::vector<std::string> param_tensor_names;
  param_tensor_names.push_back(bits_wrapper.GetParamTensorName());
  param_tensor_names.push_back(block_size_wrapper.GetParamTensorName());
  param_tensor_names.push_back(K_wrapper.GetParamTensorName());
  param_tensor_names.push_back(N_wrapper.GetParamTensorName());

  LOGS(logger, INFO) << "param_tensor_names: ";
  for (const auto& name : param_tensor_names) {
    LOGS(logger, INFO) << "  - " << name;
  }

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(bits_wrapper)), "Failed to add param");
  LOGS(logger, INFO) << "Added bits param wrapper." << bits_wrapper.GetParamTensorName();
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(block_size_wrapper)), "Failed to add param");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(K_wrapper)), "Failed to add param");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(N_wrapper)), "Failed to add param");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(a_input_tensor)), "Failed to add input");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(b_input_tensor)), "Failed to add input");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scale_input_tensor)), "Failed to add input");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zeros_input_tensor)), "Failed to add input");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    "MatMulNBits",
                                                    {a_input_def.node_arg.Name(), b_input_def.node_arg.Name(), scale_input_def.node_arg.Name(), zeros_input_def.node_arg.Name()},
                                                    {output_def.node_arg.Name()},
                                                    std::move(param_tensor_names),
                                                    validate),
                    "Failed to add fused MatMulNBits fused node.");

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
