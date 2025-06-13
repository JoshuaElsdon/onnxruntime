#include "core/providers/qnn/builder/qnn_node_group/matmul_n_bits_fusion.h"

// #include "onnx/defs/schema.h"
// #include "onnx/onnx_pb.h"
#include <gsl/gsl>
#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>
#include <utility>
#include <iterator> 

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include <QnnOpDef.h>


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

  // make sure the input_dq_unit and the scale_dq_node_unit are not the same node.
  if (input_dq_unit.GetNode().Index() == scale_dq_node_unit->GetNode().Index()) {
    LOGS(logger, INFO) << "MatMulNBitsQDQFusion::TryFusion() input_dq_unit and scale_dq_node_unit are the same node.";
    return nullptr;
  } else {
    LOGS(logger, INFO) << "MatMulNBitsQDQFusion::TryFusion() input_dq_unit and scale_dq_node_unit are not the same node.";
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

onnxruntime::common::Status GetInitializerUint8TensorValues(
    const onnxruntime::GraphViewer& graph_viewer,
    const std::string& tensor_name,
    std::vector<uint8_t>& out_values,
    const onnxruntime::logging::Logger& logger) {
  
  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  LOGS(logger, INFO) << "Looking for initializer: " << tensor_name;
  if (!graph_viewer.GetInitializedTensor(tensor_name, tensor_proto)) {
    LOGS(logger, ERROR) << "Initializer not found: " << tensor_name;
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Initializer not found: ", tensor_name);
  }
  LOGS(logger, INFO) << "Found initializer: " << tensor_name;

  if (tensor_proto->dims_size() == 0) {
    LOGS(logger, ERROR) << "Initializer tensor has no dimensions: " << tensor_name;
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Initializer tensor has no dimensions: ", tensor_name);
  }

  ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(*tensor_proto, out_values));

  return onnxruntime::common::Status::OK();
}

void PrintTensorProto(const ONNX_NAMESPACE::TensorProto* tensor) {
  if (!tensor) {
    std::cout << "TensorProto: nullptr\n";
    return;
  }

  std::cout << "TensorProto:\n";

  // Name
  if (tensor->has_name()) {
    std::cout << "  Name: " << tensor->name() << "\n";
  }

  // Dims
  std::cout << "  Dims (size=" << tensor->dims_size() << "): [";
  const auto& dims = tensor->dims();
  for (int i = 0; i < dims.size(); ++i) {
    std::cout << dims[i];
    if (i + 1 < dims.size()) std::cout << ", ";
  }
  std::cout << "]\n";

  // Data type
  if (tensor->has_data_type()) {
    std::cout << "  DataType: " << tensor->data_type() << "\n";
  }

  // Data location
  if (tensor->has_data_location()) {
    std::cout << "  DataLocation: " << static_cast<int>(tensor->data_location()) << "\n";
  }

  // Raw data
  if (tensor->has_raw_data()) {
    const std::string& raw = tensor->raw_data();
    std::cout << "  RawData (size=" << raw.size() << "): ";
    size_t print_len = std::min<size_t>(16, raw.size());  // Show up to first 16 bytes
    for (size_t i = 0; i < print_len; ++i) {
      std::cout << std::hex << std::setw(2) << std::setfill('0') << (static_cast<uint8_t>(raw[i])) << " ";
    }
    if (raw.size() > print_len) std::cout << "...";
    std::cout << std::dec << "\n";
  }

  // Indicate other fields might be available (float_data, int32_data, etc.) if raw_data is not used
  std::cout << "  (Note: other typed fields like float_data/int32_data not handled in this printout)\n";
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

  // get the number of tokens from a_input_def [batch, tockens, embedding]
  int in_dims = a_input_def.node_arg.Shape()->dim_size();
  if ( in_dims < 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensor must have at least 3 dimensions for MatMulNBits fusion.");
  }
  const int64_t num_tokens = a_input_def.node_arg.Shape()->dim(in_dims-2).dim_value();

  LOGS(logger, INFO) << " node_name: " << node_name;
  LOGS(logger, INFO) << " a_input_def: " << a_input_def.node_arg.Name();
  LOGS(logger, INFO) << " num_tokens: " << num_tokens;
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

  // Get the node attributes for the MatMulNBits node.
  const Node& matmul_node = matmul_n_bits_unit.GetNode();
  const auto& matmul_node_attributes = matmul_node.GetAttributes();
  for (const auto& attr : matmul_node_attributes) {
    LOGS(logger, INFO) << "MatMulNBits node attribute: " << attr.first << " = " << attr.second.i();
    if (attr.first == "bits") {
      bits_scalar.uint32Value = static_cast<uint32_t>(attr.second.i());
    } else if (attr.first == "block_size") {
      block_size_scalar.uint32Value = static_cast<uint32_t>(attr.second.i());
    } else if (attr.first == "K") {
      K_scalar.uint32Value = static_cast<uint32_t>(attr.second.i());
    } else if (attr.first == "N") {
      N_scalar.uint32Value = static_cast<uint32_t>(attr.second.i());
    }
  }

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

  if (num_tokens == 1) {
    LOGS(logger, INFO) << "Using the MatMulNBits kernel" << validate;

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                      "MatMulNBits",
                                                      "MatMulNBits",
                                                      {a_input_def.node_arg.Name(), b_input_def.node_arg.Name(), scale_input_def.node_arg.Name(), zeros_input_def.node_arg.Name()},
                                                      {output_def.node_arg.Name()},
                                                      std::move(param_tensor_names),
                                                      validate),
                      "Failed to add fused MatMulNBits fused node.");

  } else {
    LOGS(logger, INFO) << "Using the unpack_weights kernel with regular matmul, num tokens:" << num_tokens;
    // rather than using the MatMulNBits kernel, we will use the unpack_weights kernel to get the weights, then we will pass these to a regular MatMul.

    // get the tensor values for B, scales and zeros.

    std::vector<uint8_t> b_values, zero_values, scale_values;
    ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValues(
        qnn_model_wrapper.GetGraphViewer(),
        b_input_def.node_arg.Name(),
        b_values,
        logger));
    ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValues(
        qnn_model_wrapper.GetGraphViewer(),
        zeros_input_def.node_arg.Name(),
        zero_values,
        logger));
    ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValues(
        qnn_model_wrapper.GetGraphViewer(),
        scale_input_def.node_arg.Name(),
        scale_values,
        logger));

    const Node& dq_node = scale_dq_unit.GetNode();
    const auto& input_defs = dq_node.InputDefs();

    const InitializedTensorSet& initializers = qnn_model_wrapper.GetInitializerTensors();
    // list these
    LOGS(logger, INFO) << "Initialized tensors:";
    for (const auto& initializer : initializers) {
      LOGS(logger, INFO) << "  - " << initializer.first;
      // print the first value of the initializer
      const ONNX_NAMESPACE::TensorProto* tensor_proto = initializer.second;
        if (tensor_proto->raw_data().size() >= 1) {
          LOGS(logger, INFO) << "    First value: ";
          LOGS(logger, INFO) << "      - " << static_cast<int>(tensor_proto->raw_data()[0]);  
        }
    
    }

    if (input_defs.size() >= 2) {
      const NodeArg* scale_tensor_arg = input_defs[1];  // the "scale" input
      const ONNX_NAMESPACE::TensorProto* scale_initializer = nullptr;
      if (qnn_model_wrapper.GetGraphViewer().GetInitializedTensor(scale_tensor_arg->Name(), scale_initializer)) {
        LOGS(logger, INFO) << "Found scale scale initializer: " << scale_initializer->name();
        PrintTensorProto(scale_initializer);
        float scale_val = 1.0f;
        if (scale_initializer->has_raw_data()) {
          scale_val = *reinterpret_cast<const float*>(scale_initializer->raw_data().data());
        } else {
          float data = scale_initializer->float_data(0);
          LOGS(logger, INFO) << "Using float_data: " << data;
          scale_val = data;
        }
        LOGS(logger, INFO) << "Scale value: " << scale_val;
      }
    }
    if (input_defs.size() >= 3) {
      const NodeArg* zero_tensor_arg = input_defs[2];  // the "zeros" input
      const ONNX_NAMESPACE::TensorProto* zero_initializer = nullptr;
      if (qnn_model_wrapper.GetGraphViewer().GetInitializedTensor(zero_tensor_arg->Name(), zero_initializer)) {
        LOGS(logger, INFO) << "Found zeros initializer: " << zero_initializer->name();
        PrintTensorProto(zero_initializer);
        int32_t zero_val = 100;
        if (zero_initializer->has_raw_data()) {
          zero_val = *reinterpret_cast<const float*>(zero_initializer->raw_data().data());
        } else {
          LOGS(logger, INFO) << "Using uint16_data:";
          int32_t data = zero_initializer->int32_data(0);
          LOGS(logger, INFO) << "Using uint16_data: " << data;
          zero_val = data;
        }
        LOGS(logger, INFO) << "Zero value: " << zero_val;
      }
    }

    // unpack the B tensor as 2 bit values
    std::vector<uint8_t> unpacked_b_values;
    unpacked_b_values.reserve(b_values.size() * 4);  // each 2-bit value will expand to 4 bits
    for (size_t i = 0; i < b_values.size(); ++i) {
      uint8_t byte = b_values[i];
      // Extract 4 2-bit values from the byte
      unpacked_b_values.push_back((byte >> 6) & 0x03);  // first 2 bits
      unpacked_b_values.push_back((byte >> 4) & 0x03);  // second 2 bits
      unpacked_b_values.push_back((byte >> 2) & 0x03);  // third 2 bits
      unpacked_b_values.push_back(byte & 0x03);         // last 2 bits
    }

    // print out some values
    LOGS(logger, INFO) << "Unpacked B tensor values: ";
    for (size_t i = 0; i < std::min<size_t>(unpacked_b_values.size(), 10); ++i) {  // print first 10 values
      LOGS(logger, INFO) << "  - " << static_cast<int>(unpacked_b_values[i]);
    } 
    // print out some zeros
    LOGS(logger, INFO) << "Zeros tensor values: ";
    for (size_t i = 0; i < std::min<size_t>(zero_values.size(), 10); ++i) {  // print first 10 values
      LOGS(logger, INFO) << "  - " << static_cast<int>(zero_values[i]);
    }
    // print out some scales
    LOGS(logger, INFO) << "Scale tensor values: ";  
    for (size_t i = 0; i < std::min<size_t>(scale_values.size(), 10); ++i) {  // print first 10 values
      LOGS(logger, INFO) << "  - " << static_cast<int>(scale_values[i]);
    }

    std::string weights_name = node_name + "_weights_raw";
    std::vector<uint32_t> weights_shape = {K_scalar.uint32Value, N_scalar.uint32Value};

    QnnTensorWrapper weights_tensor(weights_name,
                                    QNN_TENSOR_TYPE_NATIVE,
                                    QNN_DATATYPE_UFIXED_POINT_8,
                                    QnnQuantParamsWrapper(1.0f, 0),
                                    std::move(weights_shape));

    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weights_tensor)), "Failed to add tensor.");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                      "UnpackWeightsNBits",
                                                      "UnpackWeightsNBits",
                                                      {b_input_def.node_arg.Name(), scale_input_def.node_arg.Name(), zeros_input_def.node_arg.Name()},
                                                      {weights_name},
                                                      std::move(param_tensor_names),
                                                      validate),
                      "Failed to add fused MatMulNBits fused node.");

    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param.bool8Value = 0;
    QnnParamWrapper transpose_in0_param(input_dq_unit.Index(), input_dq_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0,
                                        scalar_param);
    std::vector<std::string> param_tensor_names;
    param_tensor_names.push_back(transpose_in0_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_in0_param));

    QnnParamWrapper transpose_in1_param(input_dq_unit.Index(), input_dq_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,
                                        scalar_param);
    param_tensor_names.push_back(transpose_in1_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_in1_param));

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name + "mat_mul", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_MAT_MUL,
                                                      {a_input_def.node_arg.Name(), weights_name}, {output_def.node_arg.Name()},
                                                      std::move(param_tensor_names), validate),
                      "Failed to add fused Matmul node.");
  }

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
