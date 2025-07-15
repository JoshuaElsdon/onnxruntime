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

static inline void split_tile_2bit(int32_t* dst,
                                   const int32_t* src,
                                   const int32_t W,
                                   const int32_t H) {
  // std::fill(dst, dst + ((H * W * 2) >> 5), 0);

  for (int32_t y = 0; y < H; ++y) {
    for (int32_t x = 0; x < W; ++x) {
      const int32_t idx = y * W + x;

      const uint32_t word = *(reinterpret_cast<const uint32_t*>(src) + (idx * 2) / 32);
      const uint32_t two = (word >> (idx * 2) % 32) & 0b11;
      const uint32_t b0 = two & 1u;
      const uint32_t b1 = (two >> 1u) & 1u;

      // Destination coordinates
      const int32_t bit_idx0 = (y / 128) * W * 128 + (x / 4) * 512 + (y & 127) * 4 + x % 4;
      const int32_t bit_idx1 = bit_idx0 + H * W;

      dst[bit_idx0 / 32] |= (b0 << (bit_idx0 % 32));
      dst[bit_idx1 / 32] |= (b1 << (bit_idx1 % 32));
    }
  }
}

static inline void split_transpose_2bit(int32_t* dst,
                                        const int32_t* src,
                                        const int32_t W,
                                        const int32_t H) {
  // std::fill(dst, dst + ((H * W * 2) >> 5), 0);

  for (int32_t y = 0; y < H; ++y) {
    for (int32_t x = 0; x < W; ++x) {
      const int32_t idx = y * W + x;

      const uint32_t word = *(reinterpret_cast<const uint32_t*>(src) + (idx * 2) / 32);
      const uint32_t two = (word >> (idx * 2) % 32) & 0b11;
      const uint32_t b0 = two & 1u;
      const uint32_t b1 = (two >> 1u) & 1u;

      // Destination coordinates
      const int32_t bit_idx0 = x * H + y;
      const int32_t bit_idx1 = bit_idx0 + H * W;

      dst[bit_idx0 / 32] |= (b0 << (bit_idx0 % 32));
      dst[bit_idx1 / 32] |= (b1 << (bit_idx1 % 32));
    }
  }
}

static inline void transpose(uint16_t* dst,
                             const uint16_t* src,
                             const int32_t W,
                             const int32_t H) {
  for (int32_t y = 0; y < H; ++y) {
    for (int32_t x = 0; x < W; ++x) {
      const int32_t src_idx = y * W + x;
      const int32_t dst_idx = x * H + y;

      dst[dst_idx] = src[src_idx];
    }
  }
}

struct KernelParams {
  Qnn_Scalar_t bits;
  Qnn_Scalar_t block;
  Qnn_Scalar_t K;
  Qnn_Scalar_t N;
};

struct TensorHandles {
  NodeUnitIODef A, B, Scales, Zeros, Out;
  std::string node_name;
  TensorHandles(const NodeUnitIODef& a,
                const NodeUnitIODef& b,
                const NodeUnitIODef& s,
                const NodeUnitIODef& z,
                const NodeUnitIODef& o,
                std::string name)
      : A(a), B(b), Scales(s), Zeros(z), Out(o), node_name(std::move(name)) {}
};

KernelParams GetKernelParams(const NodeUnit& matmul_n_bits_unit, const logging::Logger& logger) {
  KernelParams params;
  params.bits.dataType = QNN_DATATYPE_INT_32;
  params.block.dataType = QNN_DATATYPE_INT_32;
  params.K.dataType = QNN_DATATYPE_INT_32;
  params.N.dataType = QNN_DATATYPE_INT_32;

  const Node& matmul_node = matmul_n_bits_unit.GetNode();
  const auto& matmul_node_attributes = matmul_node.GetAttributes();

  for (const auto& attr : matmul_node_attributes) {
    LOGS(logger, INFO) << "MatMulNBits node attribute: " << attr.first << " = " << attr.second.i();
    if (attr.first == "bits") {
      params.bits.uint32Value = static_cast<uint32_t>(attr.second.i());
    } else if (attr.first == "block_size") {
      params.block.uint32Value = static_cast<uint32_t>(attr.second.i());
    } else if (attr.first == "K") {
      params.K.uint32Value = static_cast<uint32_t>(attr.second.i());
    } else if (attr.first == "N") {
      params.N.uint32Value = static_cast<uint32_t>(attr.second.i());
    }
  }

  // get number of input dims and output dims
  uint32_t input_dims = matmul_n_bits_unit.Inputs()[0].node_arg.Shape()->dim_size();
  uint32_t output_dims = matmul_n_bits_unit.Outputs()[0].node_arg.Shape()->dim_size();

  uint32_t k_from_input = matmul_n_bits_unit.Inputs()[0].node_arg.Shape()->dim(input_dims - 1).dim_value();
  uint32_t n_from_output = matmul_n_bits_unit.Outputs()[0].node_arg.Shape()->dim(output_dims - 1).dim_value();
  if (params.K.uint32Value != k_from_input) {
    LOGS(logger, ERROR) << "K value from MatMulNBits node attribute does not match input shape. Expected: " << k_from_input << ", got: " << params.K.uint32Value;
    throw std::invalid_argument("K value mismatch in MatMulNBits node.");
  }
  if (params.N.uint32Value != n_from_output) {
    LOGS(logger, ERROR) << "N value from MatMulNBits node attribute does not match output shape. Expected: " << n_from_output << ", got: " << params.N.uint32Value;
    throw std::invalid_argument("N value mismatch in MatMulNBits node.");
  }

  return params;
}

std::vector<std::string> load_parmams_to_qnn(QnnModelWrapper& qnn_model_wrapper, const NodeIndex& index, const KernelParams& kernel_params, const TensorHandles& handles, const std::string& extra_name) {
  QnnParamWrapper bits_wrapper(index, handles.node_name + extra_name, "bits", kernel_params.bits);
  QnnParamWrapper block_size_wrapper(index, handles.node_name + extra_name, "block_size", kernel_params.block);
  QnnParamWrapper K_wrapper(index, handles.node_name + extra_name, "K", kernel_params.K);
  QnnParamWrapper N_wrapper(index, handles.node_name + extra_name, "N", kernel_params.N);
  std::vector<std::string> param_tensor_names;
  param_tensor_names.push_back(bits_wrapper.GetParamTensorName());
  param_tensor_names.push_back(block_size_wrapper.GetParamTensorName());
  param_tensor_names.push_back(K_wrapper.GetParamTensorName());
  param_tensor_names.push_back(N_wrapper.GetParamTensorName());

  qnn_model_wrapper.AddParamWrapper(std::move(bits_wrapper));
  qnn_model_wrapper.AddParamWrapper(std::move(block_size_wrapper));
  qnn_model_wrapper.AddParamWrapper(std::move(K_wrapper));
  qnn_model_wrapper.AddParamWrapper(std::move(N_wrapper));

  return param_tensor_names;
}

TensorHandles GetTensorHandles(const NodeUnit& input_dq_unit, const NodeUnit& matmul_n_bits_unit, const NodeUnit& output_q_unit, const NodeUnit& scale_dq_unit) {
  TensorHandles handles(
      input_dq_unit.Inputs()[0],
      matmul_n_bits_unit.Inputs()[1],
      scale_dq_unit.Inputs()[0],
      matmul_n_bits_unit.Inputs()[3],
      output_q_unit.Outputs()[0],
      utils::GetNodeName(input_dq_unit));

  return handles;
}

struct ParsedHints {
  bool shuffle = false;  // true ⇒ use fast shuffle kernel
  bool scratch = false;  // true ⇒ use scratch memory for fast shuffle kernel
  bool split = false;
  uint32_t split_size = 0;   // 0 ⇒ none
  uint32_t split_count = 1;  // 1 ⇒ no split, 2 ⇒ split into two tensors, etc.
};

ParsedHints parse_hints(QnnModelWrapper& qnn_model_wrapper, const int output_dimension, const logging::Logger& logger) {
  ParsedHints hints;
  const ModelSettings& model_settings = qnn_model_wrapper.GetModelSettings();
  const std::string& model_hints = model_settings.model_hints;

  LOGS(logger, INFO) << "Model hints: " << model_hints;

  if (model_hints.find("shuffle") != std::string::npos) {
    hints.shuffle = true;
    LOGS(logger, INFO) << "Model hint 'shuffle' found.";
  }
  if (model_hints.find("scratch") != std::string::npos) {
    hints.scratch = true;
    LOGS(logger, INFO) << "Model hint 'scratch' found.";
  }
  if (model_hints.find("split") != std::string::npos) {
    hints.split = true;
    size_t split_pos = model_hints.find("split");
    if (split_pos != std::string::npos) {
      size_t next_underscore = model_hints.find('_', split_pos + 5);
      if (next_underscore != std::string::npos) {
        std::string split_size_str = model_hints.substr(split_pos + 5, next_underscore - (split_pos + 5));
        try {
          hints.split_size = std::stoul(split_size_str);
          LOGS(logger, INFO) << "Target out split size set to: " << hints.split_size;
        } catch (const std::invalid_argument& e) {
          LOGS(logger, ERROR) << "Invalid split size: " << split_size_str;
        }
      } else {
        LOGS(logger, ERROR) << "No underscore found after 'split'";
      }
    }
  }

  if (hints.split_size > 0) {
    // Calculate the split count based on the output dimension.
    if (output_dimension % hints.split_size != 0) {
      LOGS(logger, ERROR) << "Output dimension is not divisible by split size.";
      throw std::invalid_argument("Output dimension is not divisible by split size.");
    }
    hints.split_count = output_dimension / hints.split_size;
    LOGS(logger, INFO) << "Split count set to: " << hints.split_count;
  } else {
    // set split size tot the N dimension.
    hints.split_size = output_dimension;
    LOGS(logger, INFO) << "Split size set to output dimension: " << hints.split_size;
  }

  return hints;
}

void get_scale_quant_params(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& scale_dq_unit, float& scale_scale, int32_t& scale_zero, const logging::Logger& logger) {
  // Get the scale and zero point from the DequantizeLinear node.
  const Node& dq_node = scale_dq_unit.GetNode();
  const auto& input_defs = dq_node.InputDefs();

  if (input_defs.size() >= 2) {
    const NodeArg* scale_tensor_arg = input_defs[1];  // the "scale" input
    const ONNX_NAMESPACE::TensorProto* scale_initializer = nullptr;
    if (qnn_model_wrapper.GetGraphViewer().GetInitializedTensor(scale_tensor_arg->Name(), scale_initializer)) {
      LOGS(logger, INFO) << "Found scale initializer: " << scale_initializer->name();
      if (scale_initializer->has_raw_data()) {
        scale_scale = *reinterpret_cast<const float*>(scale_initializer->raw_data().data());
      } else {
        float data = scale_initializer->float_data(0);
        LOGS(logger, INFO) << "Using float_data: " << data;
        scale_scale = data;
      }
      LOGS(logger, INFO) << "Scale value: " << scale_scale;
    }
  }
  if (input_defs.size() >= 3) {
    const NodeArg* zero_tensor_arg = input_defs[2];  // the "zeros" input
    const ONNX_NAMESPACE::TensorProto* zero_initializer = nullptr;
    if (qnn_model_wrapper.GetGraphViewer().GetInitializedTensor(zero_tensor_arg->Name(), zero_initializer)) {
      LOGS(logger, INFO) << "Found zeros initializer: " << zero_initializer->name();
      if (zero_initializer->has_raw_data()) {
        scale_zero = *reinterpret_cast<const int32_t*>(zero_initializer->raw_data().data());
      } else {
        LOGS(logger, INFO) << "Using uint16_data:";
        int32_t data = zero_initializer->int32_data(0);
        LOGS(logger, INFO) << "Using uint16_data: " << data;
        scale_zero = data;
      }
      LOGS(logger, INFO) << "Zero value: " << scale_zero;
    }
  }
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

  const TensorHandles& handles = GetTensorHandles(input_dq_unit, matmul_n_bits_unit, output_q_unit, scale_dq_unit);

  // bits, block size, K and N.
  KernelParams kernel_params = GetKernelParams(matmul_n_bits_unit, logger);
  // get the hints, shuffle, scratch and split size etc.
  ParsedHints hints = parse_hints(qnn_model_wrapper, kernel_params.N.uint32Value, logger);

  uint32_t in_dims = handles.A.node_arg.Shape()->dim_size();
  if (in_dims < 3) {
    throw std::invalid_argument("Input tensor must have at least 3 dimensions for MatMulNBits fusion.");
  }
  uint32_t token_count = handles.A.node_arg.Shape()->dim(in_dims - 2).dim_value();

  std::vector<std::string> split_b_tensor_names;
  std::vector<std::string> split_scales_tensor_names;
  std::vector<std::string> split_zeros_tensor_names;
  std::vector<std::string> split_output_tensor_names;  // this will contain the final output if split == 1, otherwise it will contain intermediate outputs that should be concatenated.

  // get the original tensor values for B, scales and zeros.
  std::vector<uint8_t> b_values_orig, zero_values_orig, scale_values_orig;
  ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValues(
      qnn_model_wrapper.GetGraphViewer(),
      handles.B.node_arg.Name(),
      b_values_orig,
      logger));
  ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValues(
      qnn_model_wrapper.GetGraphViewer(),
      handles.Scales.node_arg.Name(),
      scale_values_orig,
      logger));
  ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValues(
      qnn_model_wrapper.GetGraphViewer(),
      handles.Zeros.node_arg.Name(),
      zero_values_orig,
      logger));

  // These tensors exist in every option.
  QnnTensorWrapper a_input_tensor, output_tensor;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(handles.A, a_input_tensor));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(handles.Out, output_tensor));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(a_input_tensor)), "Failed to add input");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");

  // get the output tensor information
  if (hints.split_count > 1) {
    for (size_t i = 0; i < hints.split_count; ++i) {
        TensorInfo output_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(handles.Out, output_info));
      output_info.shape[output_info.shape.size()-1] = hints.split_size;
      // make some output tensors.
      std::string output_name = handles.node_name + "Output_" + std::to_string(i);
      QnnTensorWrapper output_tensor(
          output_name,
          QNN_TENSOR_TYPE_NATIVE,
          output_info.qnn_data_type,
          std::move(output_info.quant_param),  // If unquantized, otherwise pass scale/offset
          std::move(output_info.shape));
      // add the tensor to the model wrapper.
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output tensor");
      split_output_tensor_names.push_back(output_name);
      LOGS(logger, INFO) << "Added output tensor: " << output_name << " with shape_size " << output_info.shape.size();
      for (size_t j = 0; j < output_info.shape.size(); ++j) {
        LOGS(logger, INFO) << "Output tensor shape[" << j << "]: " << output_info.shape[j];
      }
      
    }
  } else {
    // if split_count is 1, we just use the original output tensor.
    split_output_tensor_names.push_back(handles.Out.node_arg.Name());
  }

  float scale_scale = 1.0f;
  int32_t scale_zero = 0;
  get_scale_quant_params(qnn_model_wrapper, scale_dq_unit, scale_scale, scale_zero, logger);

  if (!hints.shuffle) {
    // if target_out_split_size is set, we need to split the B, scales and zeros tensors.
    LOGS(logger, INFO) << "Splitting B, scales and zeros tensors into smaller chunks of size: " << hints.split_size;

    for (size_t i = 0; i < hints.split_count; ++i) {
      LOGS(logger, INFO) << "Splitting B, scales and zeros tensors into chunk: " << i;

      size_t tensor_elements = hints.split_size * kernel_params.K.uint32Value;  // each chunk has target_out_split_size*in_size elements.

      // process the B input.
      std::string b_input_name = handles.node_name + "B_" + std::to_string(i);

      // make a vector of vectors of size target_out_split_size.
      size_t b_chunk_size = tensor_elements / 4;  // each chunk has target_out_split_size*in_size elements, 4 are packed into a byte.
      // print the b_chunk_size
      LOGS(logger, INFO) << "B chunk size: " << b_chunk_size;
      // split the b_values into chunks of size b_chunk_size.
      std::vector<uint8_t> b_values_split(b_values_orig.begin() + i * b_chunk_size, b_values_orig.begin() + (i + 1) * b_chunk_size);
      TensorInfo b_info = {};
      // print the original tensor info
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(handles.B, b_info));
      // update the shape to reflect the split size
      b_info.shape[0] = hints.split_size;  // update the shape to reflect the split size

      QnnTensorWrapper b_input_tensor(
          b_input_name,
          QNN_TENSOR_TYPE_STATIC,  // It's an initializer
          QNN_DATATYPE_UINT_8,
          std::move(b_info.quant_param),  // If unquantized, otherwise pass scale/offset
          std::move(b_info.shape),
          std::move(b_values_split)  // your replacement buffer
      );
      // add the tensor to the model wrapper.
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(b_input_tensor)), "Failed to add input");
      split_b_tensor_names.push_back(b_input_name);

      // process the scale input.
      std::string scale_input_name = handles.node_name + "Scale_" + std::to_string(i);
      // make a vector of vectors of size target_out_split_size.
      size_t scale_chunk_size = 2 * tensor_elements / 64;  // each chunk has target_out_split_size*in_size elements/ 64 elements, they are in a 16-bit format.
      std::vector<uint8_t> scale_values_split(scale_values_orig.begin() + i * scale_chunk_size, scale_values_orig.begin() + (i + 1) * scale_chunk_size);
      TensorInfo scale_info = {};
      // print the original tensor info
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(handles.Scales, scale_info));
      // get the number of dims
      [[maybe_unused]] size_t scale_num_dims = scale_info.shape.size();
      // print the shape
      scale_info.shape[0] = hints.split_size;  // update the shape to reflect the split size
      QnnTensorWrapper scale_input_tensor(
          scale_input_name,
          QNN_TENSOR_TYPE_STATIC,  // It's an initializer
          QNN_DATATYPE_UFIXED_POINT_16,
          scale_info.quant_param.Copy(),  // If unquantized, otherwise pass scale/offset
          std::move(scale_info.shape),
          std::move(scale_values_split)  // your replacement buffer
      );
      // add the tensor to the model wrapper
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scale_input_tensor)), "Failed to add input");
      split_scales_tensor_names.push_back(scale_input_name);

      // process the zeros input.
      std::string zeros_input_name = handles.node_name + "Zeros_" + std::to_string(i);
      // make a vector of vectors of size target_out_split_size.
      size_t zeros_chunk_size = tensor_elements / (64 * 4);  // each chunk has target_out_split_size*in_size/64 elements, 4 are packed into a byte.
      std::vector<uint8_t> zeros_values_split(zero_values_orig.begin() + i * zeros_chunk_size, zero_values_orig.begin() + (i + 1) * zeros_chunk_size);
      TensorInfo zeros_info = {};
      // print the original tensor info
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(handles.Zeros, zeros_info));
      zeros_info.shape[0] = hints.split_size;  // update the shape to reflect the split size
      QnnTensorWrapper zeros_input_tensor(
          zeros_input_name,
          QNN_TENSOR_TYPE_STATIC,  // It's an initializer
          QNN_DATATYPE_UINT_8,
          std::move(zeros_info.quant_param),  // If unquantized, otherwise pass scale/offset
          std::move(zeros_info.shape),
          std::move(zeros_values_split)  // your replacement buffer
      );
      // LOGS(logger, INFO) << "Created Zeros input tensor: " << zeros_input_name << " with shape: " << zeros_info.shape;
      // add the tensor to the model wrapper
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zeros_input_tensor)), "Failed to add input");
      split_zeros_tensor_names.push_back(zeros_input_name);
    }

  }

  else {  // hints.shuffle is true
    LOGS(logger, INFO) << "Model hints is 'shuffle'.";
    // here we modify the input tensors for B, scales and zeros to be shuffled versions of the original tensors.
    // using teh split_tile_2bit, split_transpose_2bit and transpose functions to create the shuffled tensors.

    size_t tensor_elements = hints.split_size * kernel_params.K.uint32Value;

    for (size_t i = 0; i < hints.split_count; ++i) {
      LOGS(logger, INFO) << "Shuffling B, scales and zeros tensors into chunk: " << i;

      std::vector<uint8_t> b_values;
      std::string b_split_name = handles.node_name + "B_Shuffled" + std::to_string(i);
      LOGS(logger, INFO) << "Processing B input: " << b_split_name;

      // get the subset of the original B values for the current chunk.
      size_t b_chunk_size = tensor_elements / 4;  // each chunk has target_out_split_size*in_size elements, 4 are packed into a byte.
      LOGS(logger, INFO) << "B chunk size: " << b_chunk_size;
      // split the b_values into chunks of size b_chunk_size.
      b_values.assign(b_values_orig.begin() + i * b_chunk_size, b_values_orig.begin() + (i + 1) * b_chunk_size);

      // ensure allignment of b_values to 32 bits
      std::vector<int32_t> b_values_shuff_32(b_values.size() / sizeof(int32_t), 0);
      split_tile_2bit(b_values_shuff_32.data(), reinterpret_cast<int32_t*>(b_values.data()), kernel_params.K.uint32Value, hints.split_size);
      uint8_t* bytes = reinterpret_cast<uint8_t*>(b_values_shuff_32.data());
      std::vector<uint8_t> b_values_shuff(bytes, bytes + b_values.size());

      TensorInfo b_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(handles.B, b_info));
      b_info.shape = {1, 2, kernel_params.K.uint32Value / 8, hints.split_size};  // reshape to 1, 2, N, K/block_size
      QnnTensorWrapper b_tensor_wrapper(
          b_split_name,
          QNN_TENSOR_TYPE_STATIC,  // It's an initializer
          QNN_DATATYPE_UINT_8,
          std::move(b_info.quant_param),
          std::move(b_info.shape),
          std::move(b_values_shuff));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(b_tensor_wrapper)), "Failed to add shuffled B tensor");
      split_b_tensor_names.push_back(b_split_name);

      std::vector<uint8_t> scale_values;
      std::string scale_input_name = handles.node_name + "Scale_Shuffled" + std::to_string(i);
      LOGS(logger, INFO) << "Processing scale input: " << scale_input_name;

      // get the subset of the original scale values for the current chunk.
      size_t scale_chunk_size = 2 * tensor_elements / 64;  // each chunk has target_out_split_size*in_size elements/ 64 elements, they are in a 16-bit format.
      LOGS(logger, INFO) << "Scale chunk size: " << scale_chunk_size;
      // split the scale_values into chunks of size scale_chunk_size.
      scale_values.assign(scale_values_orig.begin() + i * scale_chunk_size, scale_values_orig.begin() + (i + 1) * scale_chunk_size);

      // ensure allignment of scale_values to 16 bits
      std::vector<uint16_t> scale_values_shuff_16(scale_values.size() / sizeof(uint16_t), 0);
      transpose(scale_values_shuff_16.data(), reinterpret_cast<uint16_t*>(scale_values.data()), kernel_params.K.uint32Value / kernel_params.block.uint32Value, hints.split_size);

      uint8_t* scale_bytes = reinterpret_cast<uint8_t*>(scale_values_shuff_16.data());
      std::vector<uint8_t> scale_values_shuff(scale_bytes, scale_bytes + scale_values.size());

      TensorInfo scales_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(handles.Scales, scales_info));
      scales_info.shape = {1, 1, hints.split_size, kernel_params.K.uint32Value / (kernel_params.block.uint32Value)};  // reshape to 1, 2, N, K/block_size
      QnnTensorWrapper scale_tensor_wrapper(
          scale_input_name,
          QNN_TENSOR_TYPE_STATIC,  // It's an initializer
          QNN_DATATYPE_UFIXED_POINT_16,
          scales_info.quant_param.Copy(),
          std::move(scales_info.shape),
          std::move(scale_values_shuff));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scale_tensor_wrapper)), "Failed to add shuffled scale tensor");
      split_scales_tensor_names.push_back(scale_input_name);

      std::vector<uint8_t> zero_values;
      std::string zeros_input_name = handles.node_name + "Zeros_Shuffled" + std::to_string(i);
      LOGS(logger, INFO) << "Processing zeros input: " << zeros_input_name;

      // get the subset of the original zeros values for the current chunk.
      size_t zeros_chunk_size = tensor_elements / (64 * 4);  // each chunk has target_out_split_size*in_size/64 elements, 4 are packed into a byte.
      LOGS(logger, INFO) << "Zeros chunk size: " << zeros_chunk_size;
      // split the zero_values into chunks of size zeros_chunk_size.
      zero_values.assign(zero_values_orig.begin() + i * zeros_chunk_size, zero_values_orig.begin() + (i + 1) * zeros_chunk_size);

      // ensure allignment of zero_values to 32 bits
      std::vector<int32_t> zero_values_shuff_32(zero_values.size() / sizeof(int32_t), 0);
      split_transpose_2bit(zero_values_shuff_32.data(), reinterpret_cast<int32_t*>(zero_values.data()), kernel_params.K.uint32Value / kernel_params.block.uint32Value, hints.split_size);

      uint8_t* zero_bytes = reinterpret_cast<uint8_t*>(zero_values_shuff_32.data());
      std::vector<uint8_t> zero_values_shuff(zero_bytes, zero_bytes + zero_values.size());

      TensorInfo zero_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(handles.Zeros, zero_info));
      zero_info.shape = {1, 2, hints.split_size, kernel_params.K.uint32Value / (kernel_params.block.uint32Value * 8)};  // reshape to 1, 2, K/block_size, N
      QnnTensorWrapper zeros_tensor_wrapper(
          zeros_input_name,
          QNN_TENSOR_TYPE_STATIC,  // It's an initializer
          QNN_DATATYPE_UINT_8,
          std::move(zero_info.quant_param),
          std::move(zero_info.shape),
          std::move(zero_values_shuff));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zeros_tensor_wrapper)), "Failed to add shuffled zeros tensor");
      split_zeros_tensor_names.push_back(zeros_input_name);
    }
  }

  if (token_count == 1) {
    LOGS(logger, INFO) << "Using the MatMulNBits kernel" << validate;

    LOGS(logger, INFO) << "Making scratch buffer " << validate;
    for (size_t i = 0; i < hints.split_count; ++i) {
      std::vector<std::string> param_tensor_names = load_parmams_to_qnn(qnn_model_wrapper, matmul_n_bits_unit.Index(), kernel_params, handles, "_split_" + std::to_string(i));

      if (hints.scratch) {
        // scratch buffer sizes, maybe move inside a class
        uint32_t SCALES_COUNT = hints.split_size * kernel_params.K.uint32Value / kernel_params.block.uint32Value;
        int32_t GROUP_SIZE = 4;
        int32_t LUT_WIDTH = 2 << (GROUP_SIZE - 1);

        size_t x_data_fp_size = kernel_params.K.uint32Value * sizeof(uint16_t);  // same size as Float16
        size_t scales_data_fp_size = SCALES_COUNT * sizeof(uint16_t);
        size_t result_size = hints.split_size * sizeof(float);
        size_t bit_sum_size = kernel_params.bits.uint32Value * hints.split_size * sizeof(uint32_t);
        size_t lut_size = (kernel_params.K.uint32Value / GROUP_SIZE) * LUT_WIDTH * sizeof(uint16_t);
        size_t offset_size = (kernel_params.K.uint32Value / kernel_params.block.uint32Value) * sizeof(uint16_t);

        size_t scratch_size = x_data_fp_size + scales_data_fp_size + result_size + bit_sum_size + lut_size + offset_size;

        // scratch shape
        std::vector<uint32_t> scratch_shape = {1, 1, 1, (uint32_t)scratch_size};  // This is a placeholder, actual shape will be determined by the kernel.
        std::string scratch_name = handles.node_name + "Scratch_" + std::to_string(i);
        QnnTensorWrapper scratch_tensor_wrapper(
            scratch_name,
            QNN_TENSOR_TYPE_NATIVE,
            QNN_DATATYPE_UINT_8,
            std::move(QnnQuantParamsWrapper()),  // If unquantized, otherwise pass scale/offset
            std::move(scratch_shape));
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scratch_tensor_wrapper)), "Failed to add scratch tensor");

        ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(handles.node_name + "_split_" + std::to_string(i),
                                                          "MatMulNBits",
                                                          "MatMulNBits",
                                                          {handles.A.node_arg.Name(), split_b_tensor_names[i], split_scales_tensor_names[i], split_zeros_tensor_names[i]},
                                                          {split_output_tensor_names[i], scratch_name},
                                                          std::move(param_tensor_names),
                                                          validate),
                          "Failed to add fused MatMulNBits fused node.");

      } else {  // hints.scratch = false
        LOGS(logger, INFO) << "Using the MatMulNBits kernel without scratch buffer";
        ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(handles.node_name + "_split_" + std::to_string(i),
                                                          "MatMulNBits",
                                                          "MatMulNBits",
                                                          {handles.A.node_arg.Name(), split_b_tensor_names[i], split_scales_tensor_names[i], split_zeros_tensor_names[i]},
                                                          {split_output_tensor_names[i]},
                                                          std::move(param_tensor_names),
                                                          validate),
                          "Failed to add fused MatMulNBits fused node without scratch buffer.");
      }
    }

    if (hints.split_count != 1) {
      std::vector<std::string> param_tensor_names_concat;
      int output_ndim = handles.Out.node_arg.Shape()->dim_size();
      int32_t default_axis = output_ndim - 1;
      Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
      axis_qnn_scalar.dataType = QNN_DATATYPE_INT_32;
      axis_qnn_scalar.int32Value = default_axis;
      QnnParamWrapper axis_param(input_dq_unit.Index(), input_dq_unit.Name(), QNN_OP_SOFTMAX_PARAM_AXIS, axis_qnn_scalar);
      param_tensor_names_concat.push_back(axis_param.GetParamTensorName());
      qnn_model_wrapper.AddParamWrapper(std::move(axis_param));
      // if we are splitting the output, we need to concatenate the outputs.
      std::string concat_name = handles.node_name + "Concat";
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(concat_name,
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        QNN_OP_CONCAT,
                                                        std::move(split_output_tensor_names),
                                                        {handles.Out.node_arg.Name()},
                                                        std::move(param_tensor_names_concat),
                                                        validate),
                        "Failed to add Concat node.");
    }

  } else { // token_count > 1
    LOGS(logger, INFO) << "Using the unpack_weights kernel with regular matmul, num tokens:" << token_count;
    // rather than using the MatMulNBits kernel, we will use the unpack_weights kernel to get the weights, then we will pass these to a regular MatMul.

    // unpack the B tensor as 2 bit values
    std::vector<uint8_t> unpacked_b_values;
    unpacked_b_values.reserve(b_values_orig.size() * 4);  // each 2-bit value will expand to 4 bits
    for (size_t i = 0; i < b_values_orig.size(); ++i) {
      uint8_t byte = b_values_orig[i];
      // Extract 4 2-bit values from the byte
      unpacked_b_values.push_back(byte & 0x03);         // last 2 bits
      unpacked_b_values.push_back((byte >> 2) & 0x03);  // third 2 bits
      unpacked_b_values.push_back((byte >> 4) & 0x03);  // second 2 bits
      unpacked_b_values.push_back((byte >> 6) & 0x03);  // first 2 bits
    }

    // unpack the zeros tensor as 2 bit values
    std::vector<uint8_t> unpacked_zeros_values;
    unpacked_zeros_values.reserve(zero_values_orig.size() * 4);  // each 2-bit value will expand to 4 bits
    for (size_t i = 0; i < zero_values_orig.size(); ++i) {
      uint8_t byte = zero_values_orig[i];
      // Extract 4 2-bit values from the byte
      unpacked_zeros_values.push_back(byte & 0x03);         // last 2 bits
      unpacked_zeros_values.push_back((byte >> 2) & 0x03);  // third 2 bits
      unpacked_zeros_values.push_back((byte >> 4) & 0x03);  // second 2 bits
      unpacked_zeros_values.push_back((byte >> 6) & 0x03);  // first 2 bits
    }

    // get the scales
    std::vector<float> unpacked_scales;
    unpacked_scales.reserve(scale_values_orig.size());
    for (size_t i = 0; i < scale_values_orig.size(); i = i + 2) {
      // Convert each uint8_t scale value to float
      unpacked_scales.push_back(static_cast<float>(scale_values_orig[i] + scale_values_orig[i + 1] * 256 - scale_zero) * scale_scale);  // Assuming scale is in [0, 255]
    }

    // loop through all the weights in batches of 64 and subtract a zero value and scale the value
    // for logging and debugging purposes we get all the floating points numbers in a vector,
    // we could have just extracted the min and max as we went along, but this is easier to debug.
    // TODO , optimize when the functionality is confirmed to work.
    std::vector<float> weights_float;
    weights_float.reserve(unpacked_b_values.size());  // reserve enough space for the weights
    for (size_t group = 0; group < unpacked_b_values.size() / 64; ++group) {
      for (size_t j = 0; j < 64; ++j) {
        size_t index = group * 64 + j;
        if (index < unpacked_b_values.size()) {
          // Get the 2-bit value, subtract the zero value, and scale it
          float weight_value = static_cast<float>(unpacked_b_values[index] - unpacked_zeros_values[group]) * unpacked_scales[group];
          weights_float.push_back(weight_value);
        }
      }
    }

    // get the min and max of the weights
    float min_weight = std::numeric_limits<float>::max();
    float max_weight = std::numeric_limits<float>::lowest();
    for (const auto& weight : weights_float) {
      if (weight < min_weight) {
        min_weight = weight;
      }
      if (weight > max_weight) {
        max_weight = weight;
      }
    }

    if (min_weight > 0.0f) {
      min_weight = 0.0f;  // Ensure min_weight is not greater than 0
    }
    if (max_weight < 0.0f) {
      max_weight = 0.0f;  // Ensure max_weight is not less than 0
    }
    if (min_weight == max_weight) {
      max_weight += 0.00001f;
    }

    // convert these to a scale and offset for a 8 bit unsigned fixed point representation
    float scale = (max_weight - min_weight) / 255.0f;  // Scale for 8-bit unsigned fixed point
    int32_t offset = -static_cast<int32_t>(std::round(-min_weight / scale));

    LOGS(logger, INFO) << "Scale: " << scale << ", Offset: " << offset;

    // now we make for loop for each of the split weights, scales and zeros tensors.
    for (size_t i = 0; i < hints.split_count; ++i) {
      LOGS(logger, INFO) << "Creating UnpackWeightsNBits node for split: " << i;
      // create the weights tensor name
      std::string weights_name = handles.node_name + "_weights_" + std::to_string(i);
      std::vector<uint32_t> weights_shape = {hints.split_size, kernel_params.K.uint32Value};
      QnnTensorWrapper weights_tensor(weights_name,
                                      QNN_TENSOR_TYPE_NATIVE,
                                      QNN_DATATYPE_UFIXED_POINT_8,
                                      QnnQuantParamsWrapper(scale, offset),
                                      std::move(weights_shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weights_tensor)), "Failed to add tensor.");

      std::vector<std::string> param_tensor_names_split = load_parmams_to_qnn(qnn_model_wrapper, matmul_n_bits_unit.Index(), kernel_params, handles, "_split_" + std::to_string(i));
      std::string unpack_name = handles.node_name + "_unpack_" + std::to_string(i);
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(unpack_name,
                                                        "UnpackWeightsNBits",
                                                        "UnpackWeightsNBits",
                                                        {split_b_tensor_names[i], split_scales_tensor_names[i], split_zeros_tensor_names[i]},
                                                        {weights_name},
                                                        std::move(param_tensor_names_split),
                                                        validate),
                        "Failed to add fused MatMulNBits fused node.");

      std::vector<std::string> param_tensor_names_mul;

      Qnn_Scalar_t t0 = QNN_SCALAR_INIT;
      t0.dataType = QNN_DATATYPE_BOOL_8;
      t0.bool8Value = 0;
      QnnParamWrapper p0(input_dq_unit.Index(), input_dq_unit.Name()+ std::to_string(i) , QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0 , t0);
      param_tensor_names_mul.push_back(p0.GetParamTensorName());
      qnn_model_wrapper.AddParamWrapper(std::move(p0));

      Qnn_Scalar_t t1 = QNN_SCALAR_INIT;
      t1.dataType = QNN_DATATYPE_BOOL_8;
      t1.bool8Value = 1;  // transpose the wieght input.
      QnnParamWrapper p1(input_dq_unit.Index(), input_dq_unit.Name()+ std::to_string(i) , QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1 ,t1);
      param_tensor_names_mul.push_back(p1.GetParamTensorName());
      qnn_model_wrapper.AddParamWrapper(std::move(p1));
      std::string matmul_op_name = handles.node_name + "_mat_mul_" + std::to_string(i);
      LOGS(logger, INFO) << "Creating MatMul node: " << matmul_op_name;
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(matmul_op_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        QNN_OP_MAT_MUL,
                                                        {handles.A.node_arg.Name(), weights_name}, {split_output_tensor_names[i]},
                                                        std::move(param_tensor_names_mul), validate),
                        "Failed to add fused Matmul node.");
    }

    if (hints.split_count > 1) {
      LOGS(logger, INFO) << "Concatenating the outputs of the MatMul nodes.";
      // now we need to add the output node, which is a concat of all the matmul outputs.
      std::vector<std::string> param_tensor_names_concat;
      int output_ndim = handles.Out.node_arg.Shape()->dim_size();
      int32_t default_axis = output_ndim - 1;
      Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
      axis_qnn_scalar.dataType = QNN_DATATYPE_INT_32;
      axis_qnn_scalar.int32Value = default_axis;
      QnnParamWrapper axis_param(input_dq_unit.Index(), input_dq_unit.Name(), QNN_OP_SOFTMAX_PARAM_AXIS, axis_qnn_scalar);
      param_tensor_names_concat.push_back(axis_param.GetParamTensorName());
      qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(handles.node_name + "_concat",
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        QNN_OP_CONCAT,
                                                        std::move(split_output_tensor_names),
                                                        {handles.Out.node_arg.Name()},
                                                        std::move(param_tensor_names_concat),
                                                        validate),
                        "Failed to add fused Concat node.");
    }
  }

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
