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

static inline void split_tile_2bit(int32_t *dst,
                                    const int32_t *src,
                                    const int32_t W,
                                    const int32_t H)
  {
    //std::fill(dst, dst + ((H * W * 2) >> 5), 0);

    for (int32_t y = 0; y < H; ++y)
    {
      for (int32_t x = 0; x < W; ++x)
      {
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

  static inline void split_transpose_2bit(int32_t *dst,
                                          const int32_t *src,
                                          const int32_t W,
                                          const int32_t H)
  {
    //std::fill(dst, dst + ((H * W * 2) >> 5), 0);

    for (int32_t y = 0; y < H; ++y)
    {
      for (int32_t x = 0; x < W; ++x)
      {
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

  static inline void transpose(uint16_t *dst,
                              const uint16_t *src,
                              const int32_t W,
                              const int32_t H)
  {
    for (int32_t y = 0; y < H; ++y)
    {
      for (int32_t x = 0; x < W; ++x)
      {
        const int32_t src_idx = y * W + x;
        const int32_t dst_idx = x * H + y;

        dst[dst_idx] = src[src_idx];
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

  // get the hints
  // print the hints
  LOGS(logger, INFO) << "Model hints: " << qnn_model_wrapper.GetModelSettings().model_hints;
  const ModelSettings model_settings = qnn_model_wrapper.GetModelSettings();
  bool is_shuffled = model_settings.model_hints.find("shuffle")!= std::string::npos;
  bool use_scratch = model_settings.model_hints.find("scratch")!= std::string::npos;

  bool is_split = model_settings.model_hints.find("split") != std::string::npos;
  uint32_t target_out_split_size = 0; 
  if (is_split) {
    LOGS(logger, INFO) << "split is enabled.";
    // find the number between split and the next _ 
    size_t split_pos = model_settings.model_hints.find("split");
    if (split_pos != std::string::npos) {
      size_t next_underscore = model_settings.model_hints.find('_', split_pos + 5);
      if (next_underscore != std::string::npos) {
        std::string split_size_str = model_settings.model_hints.substr(split_pos + 5, next_underscore - (split_pos + 5));
        try {
          target_out_split_size = std::stoul(split_size_str);
          LOGS(logger, INFO) << "target_out_split_size set to: " << target_out_split_size;
        } catch (const std::invalid_argument& e) {
          LOGS(logger, ERROR) << "Invalid split size: " << split_size_str << "";
        }
      } else {
        LOGS(logger, ERROR) << "No underscore found after 'split'";
      }
    }
  }
  std::vector<std::string> split_b_tensor_names;
  std::vector<std::string> split_scales_tensor_names;
  std::vector<std::string> split_zeros_tensor_names;

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


  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(bits_wrapper)), "Failed to add param");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(block_size_wrapper)), "Failed to add param");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(K_wrapper)), "Failed to add param");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(N_wrapper)), "Failed to add param");

  QnnTensorWrapper a_input_tensor, b_input_tensor, scale_input_tensor, zeros_input_tensor;
  QnnTensorWrapper output_tensor;

  

  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(a_input_def, a_input_tensor));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

  std::string b_input_name = b_input_def.node_arg.Name();
  std::string scale_input_name = scale_input_def.node_arg.Name();
  std::string zeros_input_name = zeros_input_def.node_arg.Name();

  float scale_scale = 1.0f;
  int32_t scale_zero = 0;

  const Node& dq_node = scale_dq_unit.GetNode();
  const auto& input_defs = dq_node.InputDefs();

  if (input_defs.size() >= 2) {
    const NodeArg* scale_tensor_arg = input_defs[1];  // the "scale" input
    const ONNX_NAMESPACE::TensorProto* scale_initializer = nullptr;
    if (qnn_model_wrapper.GetGraphViewer().GetInitializedTensor(scale_tensor_arg->Name(), scale_initializer)) {
      LOGS(logger, INFO) << "Found scale scale initializer: " << scale_initializer->name();
      //PrintTensorProto(scale_initializer);
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
      //PrintTensorProto(zero_initializer);
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


  if (!is_shuffled) {
    // use the original tensors for B, scales and zeros.
    if (target_out_split_size == 0)
    {
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(b_input_def, b_input_tensor));
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(scale_input_def, scale_input_tensor));
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(zeros_input_def, zeros_input_tensor));

      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(b_input_tensor)), "Failed to add input");
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scale_input_tensor)), "Failed to add input");
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zeros_input_tensor)), "Failed to add input");
    }
    else {
      // if target_out_split_size is set, we need to split the B, scales and zeros tensors.
      LOGS(logger, INFO) << "Splitting B, scales and zeros tensors into smaller chunks of size: " << target_out_split_size;
      // assert that N_scalar.uint32Value is divisible by target_out_split_size
      ORT_RETURN_IF_NOT(N_scalar.uint32Value % target_out_split_size == 0,
                        "N must be divisible by target_out_split_size for MatMulNBits fusion.");
      int split_number = N_scalar.uint32Value / target_out_split_size;
      int in_size = K_scalar.uint32Value;
      
      for (int i = 0; i < split_number; ++i) {
        LOGS(logger, INFO) << "Splitting B, scales and zeros tensors into chunk: " << i;

        // process the B input.
        std::string b_input_name = node_name + "B_" + std::to_string(i);
        split_b_tensor_names.push_back(b_input_name);
        // get the values of the B input tensor.
        std::vector<uint8_t> b_values;
        ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValues(
            qnn_model_wrapper.GetGraphViewer(),
            b_input_def.node_arg.Name(),
            b_values,
            logger));
        // make a vector of vectors of size target_out_split_size.
        size_t b_chunk_size = (target_out_split_size * in_size) / 4; // each chunk has target_out_split_size*in_size elements, 4 are packed into a byte.
        // print the b_chunk_size
        LOGS(logger, INFO) << "B chunk size: " << b_chunk_size;
        // split the b_values into chunks of size b_chunk_size.
        std::vector<uint8_t> b_values_split(b_values.begin() + i * b_chunk_size, b_values.begin() + (i + 1) * b_chunk_size);
        TensorInfo b_info = {};
        // print the original tensor info
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(b_input_def, b_info));
        // get the number of dims
        size_t num_dims = b_info.shape.size();
        // print the shape
        LOGS(logger, INFO) << "Original B input shape:";
        for (size_t j = 0; j < num_dims; ++j) {
          LOGS(logger, INFO) << "Dimension " << j << ": " << b_info.shape[j];
        }
        // update the shape to reflect the split size
        b_info.shape[0] = target_out_split_size; // update the shape to reflect the split size
        // print the shape 
        LOGS(logger, INFO) << "B input shape: " << b_info.shape[0] << ", " << b_info.shape[1] << ", " << b_info.shape[2];
        // print the first 10 values of the b_values_split
        LOGS(logger, INFO) << "First 10 values of b_values_split: ";
        for (size_t j = 0; j < std::min(b_values_split.size(), static_cast<size_t>(10)); ++j) {
          LOGS(logger, INFO) << static_cast<int>(b_values_split[j]);
        } 
        QnnTensorWrapper b_input_tensor(
            b_input_name,
            QNN_TENSOR_TYPE_STATIC,  // It's an initializer
            QNN_DATATYPE_UINT_8,
            std::move(b_info.quant_param), // If unquantized, otherwise pass scale/offset
            std::move(b_info.shape),
            std::move(b_values_split)  // your replacement buffer
        );
        // LOGS(logger, INFO) << "Created B input tensor: " << b_input_name << " with shape: " << b_info.shape;
        // add the tensor to the model wrapper.
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(b_input_tensor)), "Failed to add input");

        // process the scale input.
        std::string scale_input_name = node_name + "Scale_" + std::to_string(i);
        split_scales_tensor_names.push_back(scale_input_name);
        std::vector<uint8_t> scale_values;
        ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValues(
            qnn_model_wrapper.GetGraphViewer(),
            scale_input_def.node_arg.Name(),
            scale_values,
            logger));
        // make a vector of vectors of size target_out_split_size.
        size_t scale_chunk_size = 2 * (target_out_split_size * in_size) / 64; // each chunk has target_out_split_size*in_size elements/ 64 elements, they are in a 16-bit format.
        std::vector<uint8_t> scale_values_split(scale_values.begin() + i * scale_chunk_size, scale_values.begin() + (i + 1) * scale_chunk_size);
        TensorInfo scale_info = {};
        // print the original tensor info
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(scale_input_def, scale_info));
        // get the number of dims
        [[maybe_unused]] size_t scale_num_dims = scale_info.shape.size();
        // print the shape
        scale_info.shape[0] = target_out_split_size; // update the shape to reflect the split size
        QnnTensorWrapper scale_input_tensor(
            scale_input_name,
            QNN_TENSOR_TYPE_STATIC,  // It's an initializer
            QNN_DATATYPE_UFIXED_POINT_16,
            scale_info.quant_param.Copy(), // If unquantized, otherwise pass scale/offset
            std::move(scale_info.shape),
            std::move(scale_values_split)  // your replacement buffer
        );
        // LOGS(logger, INFO) << "Created Scale input tensor: " << scale_input_name << " with shape: " << scale_info.shape;
        // add the tensor to the model wrapper
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scale_input_tensor)), "Failed to add input");

        // process the zeros input.
        std::string zeros_input_name = node_name + "Zeros_" + std::to_string(i);
        split_zeros_tensor_names.push_back(zeros_input_name);
        std::vector<uint8_t> zeros_values;
        ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValues(
            qnn_model_wrapper.GetGraphViewer(),
            zeros_input_def.node_arg.Name(),
            zeros_values,
            logger));
        // make a vector of vectors of size target_out_split_size.
        size_t zeros_chunk_size = (target_out_split_size * in_size) / (64*4); // each chunk has target_out_split_size*in_size/64 elements, 4 are packed into a byte.
        std::vector<uint8_t> zeros_values_split(zeros_values.begin() + i * zeros_chunk_size, zeros_values.begin() + (i + 1) * zeros_chunk_size);
        TensorInfo zeros_info = {};
        // print the original tensor info
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(zeros_input_def, zeros_info));
        // get the number of dims
        [[maybe_unused]] size_t zeros_num_dims = zeros_info.shape.size();
        zeros_info.shape[0] = target_out_split_size; // update the shape to reflect the split size
        QnnTensorWrapper zeros_input_tensor(
            zeros_input_name,
            QNN_TENSOR_TYPE_STATIC,  // It's an initializer
            QNN_DATATYPE_UINT_8,
            std::move(zeros_info.quant_param), // If unquantized, otherwise pass scale/offset
            std::move(zeros_info.shape),
            std::move(zeros_values_split)  // your replacement buffer
        );
        // LOGS(logger, INFO) << "Created Zeros input tensor: " << zeros_input_name << " with shape: " << zeros_info.shape;
        // add the tensor to the model wrapper
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zeros_input_tensor)), "Failed to add input");

      }
    }

  }

  else if (is_shuffled) {
    LOGS(logger, INFO) << "Model hints is 'shuffle'.";
    // here we modify the input tensors for B, scales and zeros to be shuffled versions of the original tensors. 
    // using teh split_tile_2bit, split_transpose_2bit and transpose functions to create the shuffled tensors.
    std::vector<uint8_t> b_values, zero_values, scale_values;

    b_input_name = node_name+"B_Shuffled";
    scale_input_name = node_name+"Scale_Shuffled";
    zeros_input_name = node_name+"Zeros_Shuffled";

    LOGS(logger, INFO) << "Processing B input: " << b_input_name;

    // process the B input.
    ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValues(
        qnn_model_wrapper.GetGraphViewer(),
        b_input_def.node_arg.Name(),
        b_values,
        logger));

    // ensure allignment of b_values to 32 bits
    std::vector<int32_t> b_values_shuff_32(b_values.size()/sizeof(int32_t), 0);
    split_tile_2bit(b_values_shuff_32.data(), reinterpret_cast<int32_t*>(b_values.data()), K_scalar.uint32Value, N_scalar.uint32Value);
    uint8_t* bytes = reinterpret_cast<uint8_t*>(b_values_shuff_32.data());
    std::vector<uint8_t> b_values_shuff(bytes, bytes + b_values.size());

    TensorInfo b_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(b_input_def, b_info));
    b_info.shape = {1, 2, K_scalar.uint32Value / 8, N_scalar.uint32Value }; // reshape to 1, 2, N, K/block_size
    QnnTensorWrapper b_tensor_wrapper(
        b_input_name,
        QNN_TENSOR_TYPE_STATIC,  // It's an initializer
        QNN_DATATYPE_UINT_8,
        std::move(b_info.quant_param), // If unquantized, otherwise pass scale/offset
        std::move(b_info.shape),
        std::move(b_values_shuff)  // your replacement buffer
    );
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(b_tensor_wrapper)), "Failed to add shuffled B tensor");

    // process the scale input.
    ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValues(
        qnn_model_wrapper.GetGraphViewer(),
        scale_input_def.node_arg.Name(),
        scale_values,
        logger));

    // ensure allignment of scale_values to 16 bits
    std::vector<uint16_t> scale_values_shuff_16(scale_values.size()/sizeof(uint16_t), 0);
    transpose(scale_values_shuff_16.data(), reinterpret_cast<uint16_t*>(scale_values.data()), K_scalar.uint32Value / block_size_scalar.uint32Value, N_scalar.uint32Value);

    uint8_t* scale_bytes = reinterpret_cast<uint8_t*>(scale_values_shuff_16.data());
    std::vector<uint8_t> scale_values_shuff(scale_bytes, scale_bytes + scale_values.size());

    TensorInfo scales_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(scale_input_def, scales_info));
    scales_info.shape = {1, 1, N_scalar.uint32Value, K_scalar.uint32Value / (block_size_scalar.uint32Value)}; // reshape to 1, 2, N, K/block_size
    QnnTensorWrapper scale_tensor_wrapper(
        scale_input_name,
        QNN_TENSOR_TYPE_STATIC,  // It's an initializer
        QNN_DATATYPE_UFIXED_POINT_16,
        scales_info.quant_param.Copy(), // If unquantized, otherwise pass scale/offset
        std::move(scales_info.shape),
        std::move(scale_values_shuff)  // your replacement buffer
    );
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scale_tensor_wrapper)), "Failed to add shuffled scale tensor");

    // process the zeros input.
    ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValues(
        qnn_model_wrapper.GetGraphViewer(),
        zeros_input_def.node_arg.Name(),
        zero_values,
        logger));

    // ensure allignment of zero_values to 32 bits
    std::vector<int32_t> zero_values_shuff_32(zero_values.size()/sizeof(int32_t), 0);
    split_transpose_2bit(zero_values_shuff_32.data(), reinterpret_cast<int32_t*>(zero_values.data()), K_scalar.uint32Value / block_size_scalar.uint32Value, N_scalar.uint32Value);

    uint8_t* zero_bytes = reinterpret_cast<uint8_t*>(zero_values_shuff_32.data());
    std::vector<uint8_t> zero_values_shuff(zero_bytes, zero_bytes + zero_values.size());

    TensorInfo zero_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(zeros_input_def, zero_info));
    zero_info.shape = {1, 2,N_scalar.uint32Value,  K_scalar.uint32Value / (block_size_scalar.uint32Value*8)}; // reshape to 1, 2, K/block_size, N
    QnnTensorWrapper zeros_tensor_wrapper(
        zeros_input_name,
        QNN_TENSOR_TYPE_STATIC,  // It's an initializer
        QNN_DATATYPE_UINT_8,
        std::move(zero_info.quant_param), // If unquantized, otherwise pass scale/offset
        std::move(zero_info.shape),
        std::move(zero_values_shuff)  // your replacement buffer
    );
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zeros_tensor_wrapper)), "Failed to add shuffled zeros tensor");

  }

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(a_input_tensor)), "Failed to add input");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");

  if (num_tokens == 1) {
    LOGS(logger, INFO) << "Using the MatMulNBits kernel" << validate;

    LOGS(logger, INFO) << "Making scratch buffer " << validate;

    if (use_scratch) {
        // scratch buffer sizes, maybe move inside a class
    uint32_t SCALES_COUNT = N_scalar.uint32Value * K_scalar.uint32Value / block_size_scalar.uint32Value;
    int32_t GROUP_SIZE = 4;
    int32_t LUT_WIDTH = 2 << (GROUP_SIZE - 1);

    size_t x_data_fp_size = K_scalar.uint32Value *sizeof(uint16_t); // same size as Float16
    size_t scales_data_fp_size = SCALES_COUNT * sizeof(uint16_t);
    size_t result_size = N_scalar.uint32Value * sizeof(float);
    size_t bit_sum_size = bits_scalar.uint32Value * N_scalar.uint32Value * sizeof(uint32_t);
    size_t lut_size = (K_scalar.uint32Value / GROUP_SIZE) * LUT_WIDTH * sizeof(uint16_t);
    size_t offset_size = (K_scalar.uint32Value / block_size_scalar.uint32Value) * sizeof(uint16_t);

    size_t scratch_size = x_data_fp_size + scales_data_fp_size + result_size + bit_sum_size + lut_size + offset_size;

    // scratch shape
    std::vector<uint32_t> scratch_shape = {1, 1, 1, (uint32_t)scratch_size};  // This is a placeholder, actual shape will be determined by the kernel.

    QnnTensorWrapper scratch_tensor_wrapper(
        "scratch"+node_name,
        QNN_TENSOR_TYPE_NATIVE, 
        QNN_DATATYPE_UINT_8,
        std::move(QnnQuantParamsWrapper()), // If unquantized, otherwise pass scale/offset
        std::move(scratch_shape)
    );
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scratch_tensor_wrapper)), "Failed to add scratch tensor");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                      "MatMulNBits",
                                                      "MatMulNBits",
                                                      {a_input_def.node_arg.Name(), b_input_name, scale_input_name, zeros_input_name},
                                                      {output_def.node_arg.Name(), "scratch"+node_name},
                                                      std::move(param_tensor_names),
                                                      validate),
                      "Failed to add fused MatMulNBits fused node.");
    } else {
      LOGS(logger, INFO) << "Using the MatMulNBits kernel without scratch buffer";
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                        "MatMulNBits",
                                                        "MatMulNBits",
                                                        {a_input_def.node_arg.Name(), b_input_name, scale_input_name, zeros_input_name},
                                                        {output_def.node_arg.Name()},
                                                        std::move(param_tensor_names),
                                                        validate),
                        "Failed to add fused MatMulNBits fused node without scratch buffer.");
    }

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




    // unpack the B tensor as 2 bit values
    std::vector<uint8_t> unpacked_b_values;
    unpacked_b_values.reserve(b_values.size() * 4);  // each 2-bit value will expand to 4 bits
    for (size_t i = 0; i < b_values.size(); ++i) {
      uint8_t byte = b_values[i];
      // Extract 4 2-bit values from the byte
      unpacked_b_values.push_back(byte & 0x03);         // last 2 bits
      unpacked_b_values.push_back((byte >> 2) & 0x03);  // third 2 bits
      unpacked_b_values.push_back((byte >> 4) & 0x03);  // second 2 bits
      unpacked_b_values.push_back((byte >> 6) & 0x03);  // first 2 bits
    }

    // unpack the zeros tensor as 2 bit values
    std::vector<uint8_t> unpacked_zeros_values;
    unpacked_zeros_values.reserve(zero_values.size() * 4);  // each 2-bit value will expand to 4 bits
    for (size_t i = 0; i < zero_values.size(); ++i) {
      uint8_t byte = zero_values[i];
      // Extract 4 2-bit values from the byte
      unpacked_zeros_values.push_back(byte & 0x03);         // last 2 bits
      unpacked_zeros_values.push_back((byte >> 2) & 0x03);  // third 2 bits
      unpacked_zeros_values.push_back((byte >> 4) & 0x03);  // second 2 bits
      unpacked_zeros_values.push_back((byte >> 6) & 0x03);  // first 2 bits
    }

    // get the scales
    std::vector<float> unpacked_scales;
    unpacked_scales.reserve(scale_values.size());
    for (size_t i = 0; i < scale_values.size(); i=i+2) {
      // Convert each uint8_t scale value to float
      unpacked_scales.push_back(static_cast<float>(scale_values[i] + scale_values[i+1]*256 - scale_zero) * scale_scale);  // Assuming scale is in [0, 255]
    }

    // loop through all the weights in batches of 64 and subtract a zero value and scale the value
    // for logging and debugging purposes we get all the floating points numbers in a vector, 
    // we could have just extracted the min and max as we went along, but this is easier to debug.
    // TODO , optimize when the functionality is confirmed to work.
    std::vector<float> weights_float;
    weights_float.reserve(unpacked_b_values.size());  // reserve enough space for the weights
    for (size_t group = 0; group < unpacked_b_values.size() / 64; ++group)
    {
      for (size_t j = 0; j < 64; ++j) {
        size_t index = group * 64 + j;
        if (index < unpacked_b_values.size()) {
          // Get the 2-bit value, subtract the zero value, and scale it
          float weight_value = static_cast<float>(unpacked_b_values[index] - unpacked_zeros_values[group])* unpacked_scales[group];
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

    if (min_weight > 0.0f)
    {
      min_weight = 0.0f;  // Ensure min_weight is not greater than 0
    }
    if (max_weight < 0.0f)
    {
      max_weight = 0.0f;  // Ensure max_weight is not less than 0
    }
    if (min_weight == max_weight) {
      max_weight += 0.00001f;
    }

    // convert these to a scale and offset for a 8 bit unsigned fixed point representation
    float scale = (max_weight - min_weight) / 255.0f;  // Scale for 8-bit unsigned fixed point
    int32_t offset =  -static_cast<int32_t>(std::round(-min_weight / scale)); 

    LOGS(logger, INFO) << "Scale: " << scale << ", Offset: " << offset;

    // print the first 5 weights
    LOGS(logger, INFO) << "Unpacked weights (first 5): ";
    for (size_t i = 0; i < std::min<size_t>(5, weights_float.size()); ++i) {
      LOGS(logger, INFO) << "  " << i << ": " << weights_float[i];
    }

    // now we make for loop for each of the split weights, scales and zeros tensors.
    if (target_out_split_size != 0)
    {
      std::vector<std::string> matmul_out_names;
      for (size_t i = 0; i < N_scalar.uint32Value / target_out_split_size; ++i) {
        // create the weights tensor name
        std::string weights_name = node_name + "_weights_" + std::to_string(i);
        std::vector<uint32_t> weights_shape = {target_out_split_size, K_scalar.uint32Value};
        QnnTensorWrapper weights_tensor(weights_name,
                                        QNN_TENSOR_TYPE_NATIVE,
                                        QNN_DATATYPE_UFIXED_POINT_8,
                                        QnnQuantParamsWrapper(scale, offset),
                                        std::move(weights_shape));
        std::string node_string = node_name  + std::to_string(i);
        Qnn_Scalar_t split_scalar;
        split_scalar.dataType = QNN_DATATYPE_INT_32;
        split_scalar.uint32Value = target_out_split_size;
        QnnParamWrapper bits_wrapper_split(input_dq_unit.Index(), node_string, "bits", bits_scalar);
        QnnParamWrapper block_size_wrapper_split(input_dq_unit.Index(), node_string, "block_size", block_size_scalar);
        QnnParamWrapper K_wrapper_split(input_dq_unit.Index(), node_string, "K", K_scalar);
        QnnParamWrapper N_wrapper_split(input_dq_unit.Index(), node_string, "N", split_scalar);
        std::vector<std::string> param_tensor_names_split;
        param_tensor_names_split.push_back(bits_wrapper_split.GetParamTensorName());
        param_tensor_names_split.push_back(block_size_wrapper_split.GetParamTensorName());
        param_tensor_names_split.push_back(K_wrapper_split.GetParamTensorName());
        param_tensor_names_split.push_back(N_wrapper_split.GetParamTensorName());
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(bits_wrapper_split)), "Failed to add param");
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(block_size_wrapper_split)), "Failed to add param");
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(K_wrapper_split)), "Failed to add param");
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(N_wrapper_split)), "Failed to add param");


        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weights_tensor)), "Failed to add tensor.");
        std::string unpack_name = node_name + "_unpack_" + std::to_string(i);
        ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(unpack_name,
                                                  "UnpackWeightsNBits",
                                                  "UnpackWeightsNBits",
                                                  {split_b_tensor_names[i], split_scales_tensor_names[i], split_zeros_tensor_names[i]},
                                                  {weights_name},
                                                  std::move(param_tensor_names_split),
                                                  validate),
                  "Failed to add fused MatMulNBits fused node.");

        std::string matmul_out_name = node_name + "_matmul_out_" + std::to_string(i);
        matmul_out_names.push_back(matmul_out_name);
        std::vector<uint32_t> matmul_out_shape = {1, 1, target_out_split_size};
        // get the tensor info for the final output tensor.
        TensorInfo output_info = {};
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output_def, output_info));
        QnnTensorWrapper matmul_out_tensor(matmul_out_name,
                                           QNN_TENSOR_TYPE_NATIVE,
                                           output_info.qnn_data_type,
                                           output_info.quant_param.Copy(), // If unquantized, otherwise pass scale/offset
                                           std::move(matmul_out_shape));
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(matmul_out_tensor)), "Failed to add output tensor.");

        std::vector<std::string> param_tensor_names_mul;

        Qnn_Scalar_t t0 = QNN_SCALAR_INIT; 
        t0.dataType    = QNN_DATATYPE_BOOL_8; 
        t0.bool8Value  = 0;
        QnnParamWrapper p0(input_dq_unit.Index(), input_dq_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, t0);
        param_tensor_names_mul.push_back(p0.GetParamTensorName());
        qnn_model_wrapper.AddParamWrapper(std::move(p0));

        Qnn_Scalar_t t1 = QNN_SCALAR_INIT; 
        t1.dataType    = QNN_DATATYPE_BOOL_8; 
        t1.bool8Value  = 1; // transpose the wieght input.
        QnnParamWrapper p1(input_dq_unit.Index(), input_dq_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, t1);
        param_tensor_names_mul.push_back(p1.GetParamTensorName());
        qnn_model_wrapper.AddParamWrapper(std::move(p1));
        std::string matmul_op_name = node_name + "_mat_mul_" + std::to_string(i);
        ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(matmul_op_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                          QNN_OP_MAT_MUL,
                                                          {a_input_def.node_arg.Name(), weights_name}, {matmul_out_name},
                                                          std::move(param_tensor_names_mul), validate),
                          "Failed to add fused Matmul node.");

      }

      // now we need to add the output node, which is a concat of all the matmul outputs.
      std::vector<std::string> param_tensor_names_concat;
      int output_ndim = output_def.node_arg.Shape()->dim_size();
      int32_t default_axis = output_ndim-1;
      Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
      axis_qnn_scalar.dataType = QNN_DATATYPE_INT_32;
      axis_qnn_scalar.int32Value = default_axis;
      QnnParamWrapper axis_param(input_dq_unit.Index(), input_dq_unit.Name(), QNN_OP_SOFTMAX_PARAM_AXIS, axis_qnn_scalar);
      param_tensor_names_concat.push_back(axis_param.GetParamTensorName());
      qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name + "_concat",
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        QNN_OP_CONCAT,
                                                        std::move(matmul_out_names),
                                                        {output_def.node_arg.Name()},
                                                        std::move(param_tensor_names_concat),
                                                        validate),
                        "Failed to add fused Concat node.");
    }
    else{
      // unsplit method.
    std::string weights_name = node_name + "_weights_raw";

    // this uses the unpack operator to unpack the weights at runtime.
    
    std::vector<uint32_t> weights_shape = {N_scalar.uint32Value, K_scalar.uint32Value};

    bool per_channel = false;

    QnnQuantParamsWrapper weights_quant_params;
    if (per_channel) {
      // For per-channel quantization, we need to create a scale and offset for each channel.
      std::vector<float> scales(N_scalar.uint32Value, scale);
      std::vector<int32_t> offsets(N_scalar.uint32Value, -128);
      weights_quant_params = QnnQuantParamsWrapper(gsl::span<const float>(scales), gsl::span<const int32_t>(offsets), 1, false);
    } else {
      // For per-tensor quantization, we use a single scale and offset.
      weights_quant_params = QnnQuantParamsWrapper(scale, offset);
    }

    // for testing make a fake set of weights in KxN format.

    // std::vector<uint8_t> weights_data(N_scalar.uint32Value * K_scalar.uint32Value, 0);
    QnnTensorWrapper weights_tensor(weights_name,
                                    QNN_TENSOR_TYPE_NATIVE,  // QNN_TENSOR_TYPE_NATIVE is regular, QNN_TENSOR_TYPE_STATIC is for initializers
                                    QNN_DATATYPE_UFIXED_POINT_8,
                                    std::move(weights_quant_params),
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


    std::vector<std::string> param_tensor_names_mul;

    Qnn_Scalar_t t0 = QNN_SCALAR_INIT; 
    t0.dataType    = QNN_DATATYPE_BOOL_8; 
    t0.bool8Value  = 0;
    QnnParamWrapper p0(input_dq_unit.Index(), input_dq_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, t0);
    param_tensor_names_mul.push_back(p0.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(p0));

    Qnn_Scalar_t t1 = QNN_SCALAR_INIT; 
    t1.dataType    = QNN_DATATYPE_BOOL_8; 
    t1.bool8Value  = 1; // transpose the wieght input.
    QnnParamWrapper p1(input_dq_unit.Index(), input_dq_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, t1);
    param_tensor_names_mul.push_back(p1.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(p1));

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name + "mat_mul", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_MAT_MUL,
                                                      {a_input_def.node_arg.Name(), weights_name}, {output_def.node_arg.Name()},
                                                      std::move(param_tensor_names_mul), validate),
                      "Failed to add fused Matmul node.");
    }
  }

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
