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

onnxruntime::common::Status GetInitializerUint8TensorValuesMatMulNBits(
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

struct KernelParams {
  Qnn_Scalar_t bits;
  Qnn_Scalar_t block;
  Qnn_Scalar_t K;
  Qnn_Scalar_t N;
};

KernelParams GetKernelParams(const NodeUnit& node_unit, const logging::Logger& logger) {
  KernelParams params;
  params.bits.dataType = QNN_DATATYPE_INT_32;
  params.block.dataType = QNN_DATATYPE_INT_32;
  params.K.dataType = QNN_DATATYPE_INT_32;
  params.N.dataType = QNN_DATATYPE_INT_32;

  const Node& matmul_node = node_unit.GetNode();
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
  uint32_t input_dims = node_unit.Inputs()[0].node_arg.Shape()->dim_size();
  uint32_t output_dims = node_unit.Outputs()[0].node_arg.Shape()->dim_size();

  uint32_t k_from_input = static_cast<uint32_t>(node_unit.Inputs()[0].node_arg.Shape()->dim(input_dims - 1).dim_value());
  uint32_t n_from_output = static_cast<uint32_t>(node_unit.Outputs()[0].node_arg.Shape()->dim(output_dims - 1).dim_value());
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

void get_scale_quant_params(QnnModelWrapper& qnn_model_wrapper, const NodeUnitIODef& scale_io_def, float& scale_scale, int32_t& scale_zero, const logging::Logger& logger) {
  const NodeArg* scale_scale_node_arg = &scale_io_def.quant_param.value().scale;
  const ONNX_NAMESPACE::TensorProto* scale_initializer = nullptr;
  if (qnn_model_wrapper.GetGraphViewer().GetInitializedTensor(scale_scale_node_arg->Name(), scale_initializer)) {
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

  const NodeArg* scale_zero_node_arg = scale_io_def.quant_param.value().zero_point;
  const ONNX_NAMESPACE::TensorProto* zero_initializer = nullptr;
  if (qnn_model_wrapper.GetGraphViewer().GetInitializedTensor(scale_zero_node_arg->Name(), zero_initializer)) {
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

std::vector<std::string> load_parmams_to_qnn(QnnModelWrapper& qnn_model_wrapper, const NodeIndex& index, const KernelParams& kernel_params, const std::string& extra_name) {
  QnnParamWrapper bits_wrapper(index, extra_name, "bits", kernel_params.bits);
  QnnParamWrapper block_size_wrapper(index, extra_name, "block_size", kernel_params.block);
  QnnParamWrapper K_wrapper(index, extra_name, "K", kernel_params.K);
  QnnParamWrapper N_wrapper(index, extra_name, "N", kernel_params.N);
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



/*
ProcessInputs is responsible for processing the inputs of the MatMulNBits operation.
*/
Status MatMulNBitsOpBuilder::ProcessInputs([[maybe_unused]]QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]]const NodeUnit& node_unit,
                                           [[maybe_unused]]const logging::Logger& logger,
                                           [[maybe_unused]]std::vector<std::string>& input_names,
                                           [[maybe_unused]]bool do_op_validation) const {

  LOGS(logger, INFO) << "Processing inputs for MatMulNBits operation.";
  // bits, block size, K and N.
  [[maybe_unused]] KernelParams kernel_params = GetKernelParams(node_unit, logger);

  // get the handles for the inputs.
  std::vector<NodeUnitIODef> node_inputs = node_unit.Inputs();

  // get the input dimensions of the A input.
  int num_A_dims = node_inputs[0].node_arg.Shape()->dim_size();

  // get how many tokens this MatMulNBits operation will process.
  [[maybe_unused]] int64_t num_tokens = node_inputs[0].node_arg.Shape()->dim(num_A_dims - 2).dim_value();

  // get the hints, shuffle, scratch and split size etc.
  [[maybe_unused]] QnnModelWrapper::ParsedHints hints = qnn_model_wrapper.parse_hints(kernel_params.N.uint32Value, num_tokens, logger);
  LOGS(logger, INFO) << "N=" << kernel_params.N.uint32Value << ", split_size=" << hints.split_size << ", split_count=" << hints.split_count;

  std::vector<std::string> split_b_tensor_names;
  std::vector<std::string> split_scales_tensor_names;
  std::vector<std::string> split_zeros_tensor_names;
  std::vector<std::string> split_output_tensor_names;  // this will contain the final output if split == 1, otherwise it will contain intermediate outputs that should be concatenated.

  // get the original tensor values for B, scales and zeros.
  std::vector<uint8_t> b_values_orig, scale_values_orig, zero_values_orig;
  ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValuesMatMulNBits(
      qnn_model_wrapper.GetGraphViewer(),
      node_inputs[1].node_arg.Name(),
      b_values_orig,
      logger));
  ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValuesMatMulNBits(
      qnn_model_wrapper.GetGraphViewer(),
      node_inputs[2].node_arg.Name(),
      scale_values_orig,
      logger));
  ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValuesMatMulNBits(
      qnn_model_wrapper.GetGraphViewer(),
      node_inputs[3].node_arg.Name(),
      zero_values_orig,
      logger));

  // These tensors exist in every option.
  QnnTensorWrapper a_input_tensor;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(node_inputs[0], a_input_tensor));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(a_input_tensor)), "Failed to add input");

  float scale_scale = 1.0f;
  int32_t scale_zero = 0;

  ORT_RETURN_IF_NOT(node_inputs[2].quant_param.has_value(),
                    "MatMulNBits node input 2 should have quantization parameters, but found none.");

  get_scale_quant_params(qnn_model_wrapper, node_inputs[2], scale_scale, scale_zero, logger);




  if (!hints.shuffle) {
    // if target_out_split_size is set, we need to split the B, scales and zeros tensors.
    LOGS(logger, INFO) << "Splitting B, scales and zeros tensors into smaller chunks of size: " << hints.split_size;

    for (size_t i = 0; i < hints.split_count; ++i) {
      LOGS(logger, INFO) << "Splitting B, scales and zeros tensors into chunk: " << i;

      // Calculate actual split size for this chunk (may be smaller than hints.split_size for last chunk)
      size_t remaining_n = kernel_params.N.uint32Value - (i * hints.split_size);
      if (remaining_n == 0) {
        LOGS(logger, WARNING) << "Split " << i << " has remaining_n=0, skipping (split_count may be incorrectly calculated)";
        break;
      }
      size_t actual_split_size = std::min(static_cast<size_t>(hints.split_size), remaining_n);

      size_t tensor_elements = actual_split_size * kernel_params.K.uint32Value;

      // process the B input.
      std::string b_input_name = node_unit.Name() + "B_" + std::to_string(i);

      // make a vector of vectors of size target_out_split_size.
      size_t b_chunk_size = tensor_elements / 4;  // each chunk has target_out_split_size*in_size elements, 4 are packed into a byte.
      // print the b_chunk_size
      LOGS(logger, INFO) << "B chunk size: " << b_chunk_size << " (actual_split_size: " << actual_split_size << ")";
      size_t b_offset = i * (hints.split_size * kernel_params.K.uint32Value / 4);
      size_t b_end    = b_offset + b_chunk_size;

      if (b_end > b_values_orig.size()) {
        ORT_THROW("B tensor split exceeds original size");
      }

      LOGS(logger, INFO) << "Setting up split array for B";

      // split the b_values into chunks of size b_chunk_size.
      std::vector<uint8_t> b_values_split(b_values_orig.begin() + b_offset, b_values_orig.begin() + b_end);
      TensorInfo b_info = {};
      // print the original tensor info
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_inputs[1], b_info));
      // update the shape to reflect the actual split size
      b_info.shape[0] = actual_split_size;

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

      LOGS(logger, INFO) << "Created B tensor, now for the scales";

      // process the scale input.
      std::string scale_input_name = node_unit.Name() + "Scale_" + std::to_string(i);
      // make a vector of vectors of size target_out_split_size.
      size_t scale_chunk_size = 2 * tensor_elements / 64;  // each chunk has target_out_split_size*in_size elements/ 64 elements, they are in a 16-bit format.
      size_t scale_offset = i * (2 * hints.split_size * kernel_params.K.uint32Value / 64);
      size_t scale_end = scale_offset + scale_chunk_size;
      if (scale_end > scale_values_orig.size()) {
        ORT_THROW("Scale tensor split exceeds original size");
      }
      std::vector<uint8_t> scale_values_split(scale_values_orig.begin() + scale_offset, scale_values_orig.begin() + scale_end);
      TensorInfo scale_info = {};
      // print the original tensor info
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_inputs[2], scale_info));
      // get the number of dims
      [[maybe_unused]] size_t scale_num_dims = scale_info.shape.size();
      // print the shape
      scale_info.shape[0] = actual_split_size;  // update the shape to reflect the actual split size
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

      LOGS(logger, INFO) << "Done scales, now for zeroes";

      // process the zeros input.
      std::string zeros_input_name = node_unit.Name() + "Zeros_" + std::to_string(i);
      // Calculate the expected zeros size based on the actual_split_size and block_size
      // Zeros tensor is [N, zero_points_size] where zero_points_size = (k_blocks * bits + 7) / 8
      // Each split gets [actual_split_size, zero_points_size] bytes
      uint32_t k_blocks = kernel_params.K.uint32Value / kernel_params.block.uint32Value;
      size_t zero_points_size = (k_blocks * 2 + 7) / 8;  // bits per row packed into bytes
      size_t zeros_chunk_size = actual_split_size * zero_points_size;

      size_t zeros_offset = i * (hints.split_size * zero_points_size);  // Offset uses original split_size stride
      size_t zeros_end = zeros_offset + zeros_chunk_size;

      if (zeros_end > zero_values_orig.size()) {
        ORT_THROW("Zeros tensor split exceeds original size");
      }

      std::vector<uint8_t> zeros_values_split(zero_values_orig.begin() + zeros_offset,
                                               zero_values_orig.begin() + zeros_end);

      TensorInfo zeros_info = {};
      // print the original tensor info
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_inputs[3], zeros_info));
      // Set shape to [actual_split_size, zero_points_size]
      zeros_info.shape = {static_cast<uint32_t>(actual_split_size), static_cast<uint32_t>(zero_points_size)};

      LOGS(logger, INFO) << "Zeros chunk " << i << ": zeros_chunk_size=" << zeros_chunk_size
                         << ", shape=[" << zeros_info.shape[0] << ", " << zeros_info.shape[1] << "]";

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

    for (size_t i = 0; i < hints.split_count; ++i) {
      LOGS(logger, INFO) << "Shuffling B, scales and zeros tensors into chunk: " << i;

      // Calculate actual split size for this chunk (may be smaller than hints.split_size for last chunk)
      size_t remaining_n = kernel_params.N.uint32Value - (i * hints.split_size);
      if (remaining_n == 0) {
        LOGS(logger, WARNING) << "Split " << i << " has remaining_n=0, skipping (split_count may be incorrectly calculated)";
        break;
      }
      size_t actual_split_size = std::min(static_cast<size_t>(hints.split_size), remaining_n);

      size_t tensor_elements = actual_split_size * kernel_params.K.uint32Value;

      std::vector<uint8_t> b_values;
      std::string b_split_name = node_unit.Name() + "B_" + std::to_string(i);
      LOGS(logger, INFO) << "Processing B input: " << b_split_name;

      // get the subset of the original B values for the current chunk.
      size_t b_chunk_size = tensor_elements / 4;  // each chunk has target_out_split_size*in_size elements, 4 are packed into a byte.
      LOGS(logger, INFO) << "B chunk size: " << b_chunk_size << " (actual_split_size: " << actual_split_size << ")";
      // split the b_values into chunks of size b_chunk_size.
      size_t b_offset = i * (hints.split_size * kernel_params.K.uint32Value / 4);
      size_t b_end = b_offset + b_chunk_size;

      if (b_end > b_values_orig.size()) {
        ORT_THROW("B tensor shuffle split exceeds original size");
      }

      b_values.assign(b_values_orig.begin() + b_offset, b_values_orig.begin() + b_end);

      // Copy b_values to an aligned int32_t buffer for split_tile_2bit
      // b_values contains packed 2-bit weights, 4 weights per byte
      // split_tile_2bit uses tiling with 128-row tiles, so H must be padded to a multiple of 128
      // The formula (y/128)*W*128 + ... assumes tiles of 128 rows
      size_t padded_H = ((actual_split_size + 127) / 128) * 128;

      // Pad the input b_values to padded_H rows with zeros
      // Original has actual_split_size rows, each row has K elements at 2 bits each
      size_t original_bits = actual_split_size * kernel_params.K.uint32Value * 2;
      size_t padded_bits = padded_H * kernel_params.K.uint32Value * 2;
      size_t padded_bytes = (padded_bits + 7) / 8;
      size_t padded_int32s = (padded_bytes + 3) / 4;

      std::vector<int32_t> b_values_padded(padded_int32s, 0);  // Zero-initialized for padding
      // Copy original data - it's stored row-by-row at 2 bits per element
      // For actual_split_size rows, we have (actual_split_size * K * 2) bits = (actual_split_size * K / 4) bytes
      size_t original_bytes = (original_bits + 7) / 8;
      std::memcpy(b_values_padded.data(), b_values.data(), std::min(b_values.size(), original_bytes));

      // split_tile_2bit outputs 2 bit planes for padded_H*W elements
      // Output size in int32_t: (padded_H * W * 2 bits + 31) / 32
      size_t output_bits = padded_H * kernel_params.K.uint32Value * 2;
      size_t output_int32s = (output_bits + 31) / 32;
      std::vector<int32_t> b_values_shuff_32(output_int32s, 0);

      // Pass padded_H to split_tile_2bit so tiling and bit plane layout matches kernel expectations
      split_tile_2bit(b_values_shuff_32.data(), b_values_padded.data(), kernel_params.K.uint32Value, padded_H);

      LOGS(logger, INFO) << "B tiled and split with padded_H=" << padded_H;

      uint8_t* bytes = reinterpret_cast<uint8_t*>(b_values_shuff_32.data());
      size_t output_bytes = output_int32s * sizeof(int32_t);
      std::vector<uint8_t> b_values_shuff(bytes, bytes + output_bytes);
      TensorInfo b_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_inputs[1], b_info));
      // Use padded_H in shape since split_tile_2bit requires 128-row tiles and kernel expects this layout
      b_info.shape = {1, 2, kernel_params.K.uint32Value / 8, static_cast<uint32_t>(padded_H)};
      LOGS(logger, INFO) << "B info shape " << b_info.shape[0] << ", " << b_info.shape[1] << ", " << b_info.shape[2] << ", " << b_info.shape[3];
      QnnTensorWrapper b_tensor_wrapper(
          b_split_name,
          QNN_TENSOR_TYPE_STATIC,  // It's an initializer
          QNN_DATATYPE_UINT_8,
          std::move(b_info.quant_param),
          std::move(b_info.shape),
          std::move(b_values_shuff));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(b_tensor_wrapper)), "Failed to add shuffled B tensor");
      split_b_tensor_names.push_back(b_split_name);

      LOGS(logger, INFO) << "Debug gets here 4";

      std::vector<uint8_t> scale_values;
      std::string scale_input_name = node_unit.Name() + "Scale_" + std::to_string(i);
      LOGS(logger, INFO) << "Processing scale input: " << scale_input_name;

      // get the subset of the original scale values for the current chunk.
      size_t scale_chunk_size = 2 * tensor_elements / 64;  // each chunk has target_out_split_size*in_size elements/ 64 elements, they are in a 16-bit format.
      LOGS(logger, INFO) << "Scale chunk size: " << scale_chunk_size;
      // split the scale_values into chunks of size scale_chunk_size.
      size_t scale_offset = i * (2 * hints.split_size * kernel_params.K.uint32Value / 64);
      size_t scale_end = scale_offset + scale_chunk_size;

      if (scale_end > scale_values_orig.size()) {
        ORT_THROW("Scale tensor shuffle split exceeds original size");
      }

      scale_values.assign(scale_values_orig.begin() + scale_offset, scale_values_orig.begin() + scale_end);

      // ensure allignment of scale_values to 16 bits
      // After transpose: output needs padded_H rows to match kernel expectations (S_HEIGHT = OUT_CHAN)
      // Input is (actual_split_size, K/block), output is (K/block, padded_H) with zero-padding
      size_t num_scale_elements = (kernel_params.K.uint32Value / kernel_params.block.uint32Value) * padded_H;
      std::vector<uint16_t> scale_values_shuff_16(num_scale_elements, 0);  // Zero-initialize for padding

      // Transpose and pad: for each output row (K/block rows), copy actual_split_size values
      uint16_t* scale_data_16 = reinterpret_cast<uint16_t*>(scale_values.data());
      size_t rows_out = kernel_params.K.uint32Value / kernel_params.block.uint32Value;
      for (size_t out_row = 0; out_row < rows_out; ++out_row) {
        for (size_t col = 0; col < actual_split_size; ++col) {
          // Input column-major: scale_data_16[col * rows_out + out_row]
          // Output row-major: scale_values_shuff_16[out_row * padded_H + col]
          scale_values_shuff_16[out_row * padded_H + col] = scale_data_16[col * rows_out + out_row];
        }
        // Elements from actual_split_size to padded_H are already zero-initialized
      }

      uint8_t* scale_bytes = reinterpret_cast<uint8_t*>(scale_values_shuff_16.data());
      std::vector<uint8_t> scale_values_shuff(scale_bytes, scale_bytes + num_scale_elements * sizeof(uint16_t));

      TensorInfo scales_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_inputs[2], scales_info));
      scales_info.shape = {1, 1, static_cast<uint32_t>(padded_H), kernel_params.K.uint32Value / (kernel_params.block.uint32Value)};  // Use padded_H
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
      std::string zeros_input_name = node_unit.Name() + "Zeros_" + std::to_string(i);
      LOGS(logger, INFO) << "Processing zeros input: " << zeros_input_name;

      // Calculate the expected zeros size based on the actual_split_size and block_size
      // Zeros tensor is [N, zero_points_size] where zero_points_size = (k_blocks * bits + 7) / 8
      // Each split gets [actual_split_size, zero_points_size] bytes
      uint32_t k_blocks = kernel_params.K.uint32Value / kernel_params.block.uint32Value;
      size_t zero_points_size = (k_blocks * 2 + 7) / 8;  // bits per row packed into bytes
      size_t zeros_chunk_size = actual_split_size * zero_points_size;

      LOGS(logger, INFO) << "Zeros chunk size: " << zeros_chunk_size << " (actual_split_size: " << actual_split_size << ")";

      size_t zeros_offset = i * (hints.split_size * zero_points_size);  // Offset uses original split_size stride
      size_t zeros_end = zeros_offset + zeros_chunk_size;

      if (zeros_end > zero_values_orig.size()) {
        ORT_THROW("Zeros tensor shuffle split exceeds original size");
      }

      zero_values.assign(zero_values_orig.begin() + zeros_offset, zero_values_orig.begin() + zeros_end);

      // Repack zeros from row-padded format to densely packed format, then pad to padded_H
      // Input: actual_split_size rows, each with zero_points_size bytes (may have padding bits)
      // Output: k_blocks * padded_H 2-bit values packed densely (with zero-padding for extra rows)
      size_t dense_size = (k_blocks * padded_H * 2 + 7) / 8;  // Use padded_H
      std::vector<uint8_t> zero_values_dense(dense_size, 0);  // Zero-initialize for padding

      size_t output_bit_idx = 0;
      for (size_t row = 0; row < padded_H; ++row) {
        for (size_t col = 0; col < k_blocks; ++col) {
          if (row < actual_split_size) {
            // Extract 2-bit value from input
            size_t input_bit_idx = row * zero_points_size * 8 + col * 2;
            size_t input_byte_idx = input_bit_idx / 8;
            size_t input_bit_offset = input_bit_idx % 8;
            uint8_t value = (zero_values[input_byte_idx] >> input_bit_offset) & 0x3;

            // Write to densely packed output
            size_t output_byte_idx = output_bit_idx / 8;
            size_t output_bit_offset = output_bit_idx % 8;
            zero_values_dense[output_byte_idx] |= (value << output_bit_offset);
          }
          // else: already zero-initialized for padding rows
          output_bit_idx += 2;
        }
      }

      // split_transpose_2bit transposes from (H, W) to (W, H) and separates bit planes
      // Input: (padded_H, k_blocks) of 2-bit values, densely packed (with zero padding)
      // Output: (k_blocks, padded_H) in 2 bit planes
      size_t output_bits_per_plane = k_blocks * padded_H;
      size_t output_bytes_per_plane = (output_bits_per_plane + 7) / 8;
      size_t output_total_bytes = 2 * output_bytes_per_plane;

      // Allocate output buffer with correct size
      std::vector<int32_t> zero_values_shuff_32((output_total_bytes + 3) / sizeof(int32_t), 0);
      split_transpose_2bit(zero_values_shuff_32.data(), reinterpret_cast<int32_t*>(zero_values_dense.data()), kernel_params.K.uint32Value / kernel_params.block.uint32Value, padded_H);

      uint8_t* zero_bytes = reinterpret_cast<uint8_t*>(zero_values_shuff_32.data());
      std::vector<uint8_t> zero_values_shuff(zero_bytes, zero_bytes + output_total_bytes);

      TensorInfo zero_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_inputs[3], zero_info));
      // After split_transpose_2bit: shape reflects transposed dimensions (k_blocks, padded_H) with 2 bit planes
      // Each bit plane has k_blocks rows, with padded_H bits per row packed into bytes
      uint32_t bytes_per_row = (padded_H + 7) / 8;
      zero_info.shape = {1, 2, k_blocks, bytes_per_row};
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





  return Status::OK();
}

Status MatMulNBitsOpBuilder::ProcessAttributesAndOutputs([[maybe_unused]]QnnModelWrapper& qnn_model_wrapper,
                                                         [[maybe_unused]]const NodeUnit& node_unit,
                                                         [[maybe_unused]]std::vector<std::string>&& input_names,
                                                         [[maybe_unused]]const logging::Logger& logger,
                                                        [[maybe_unused]] bool do_op_validation) const {

  [[maybe_unused]] KernelParams kernel_params = GetKernelParams(node_unit, logger);
  std::vector<NodeUnitIODef> node_inputs = node_unit.Inputs();
  // get the input dimensions of the A input.
  int num_A_dims = node_inputs[0].node_arg.Shape()->dim_size();
  // get how many tokens this MatMulNBits operation will process.
  [[maybe_unused]] int64_t num_tokens = node_inputs[0].node_arg.Shape()->dim(num_A_dims - 2).dim_value();
  [[maybe_unused]] QnnModelWrapper::ParsedHints hints = qnn_model_wrapper.parse_hints(kernel_params.N.uint32Value, num_tokens, logger);
                                                            // get the output tensor information
  std::vector<NodeUnitIODef> node_outputs = node_unit.Outputs();
  ORT_RETURN_IF(node_outputs.size() != 1, "MatMulNBits node should have exactly one output, but found ", node_outputs.size());

  QnnTensorWrapper output_tensor;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(node_outputs[0], output_tensor));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");

  // get the original tensor values for B, scales and zeros.
  std::vector<uint8_t> b_values_orig, scale_values_orig, zero_values_orig;
  ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValuesMatMulNBits(
      qnn_model_wrapper.GetGraphViewer(),
      node_inputs[1].node_arg.Name(),
      b_values_orig,
      logger));
  ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValuesMatMulNBits(
      qnn_model_wrapper.GetGraphViewer(),
      node_inputs[2].node_arg.Name(),
      scale_values_orig,
      logger));
  ORT_RETURN_IF_ERROR(GetInitializerUint8TensorValuesMatMulNBits(
      qnn_model_wrapper.GetGraphViewer(),
      node_inputs[3].node_arg.Name(),
      zero_values_orig,
      logger));

  std::vector<std::string> split_output_tensor_names;
  std::vector<std::string> intermediate_concat_output_names;
  if (hints.split_count > 1) {
    if (hints.act_tile_count == 1) {
      for (size_t i = 0; i < hints.split_count; ++i) {
          TensorInfo output_info = {};
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_outputs[0], output_info));
        output_info.shape[output_info.shape.size()-1] = hints.split_size;

        if (hints.crouton)
        {
          output_info.shape[output_info.shape.size()-3] = 8;
          output_info.shape[output_info.shape.size()-2] = output_info.shape[output_info.shape.size()-2] / 8;

          // this one gets tiled
          //output_info.shape[output_info.shape.size()-2] = output_info.shape[output_info.shape.size()-2] / 8;
          LOGS(logger, INFO) << "Crouton tiling applied to output tensor split, shape: " << output_info.shape[0] << "," << output_info.shape[1] << "," << output_info.shape[2];
        }

        // make some output tensors.
        std::string output_name = node_unit.Name() + "Output_" + std::to_string(i);
        split_output_tensor_names.push_back(output_name);
        LOGS(logger, INFO) << "Added output tensor: " << output_name << " with shape_size " << output_info.shape.size();
        for (size_t j = 0; j < output_info.shape.size(); ++j) {
          LOGS(logger, INFO) << "Output tensor shape[" << j << "]: " << output_info.shape[j];
        }
        QnnTensorWrapper output_tensor_split(
            output_name,
            QNN_TENSOR_TYPE_NATIVE,
            output_info.qnn_data_type,
            std::move(output_info.quant_param),  // If unquantized, otherwise pass scale/offset
            std::move(output_info.shape));
        // add the tensor to the model wrapper.
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_split)), "Failed to add output tensor");


      }
    } else if (hints.act_tile_count > 1) {
      // if both split_count and act_tile_count are > 1, we need to create the intermediate output tensors.
      for (size_t i = 0; i < hints.split_count; ++i) {
        for (size_t j = 0; j < hints.act_tile_count; ++j) {
            TensorInfo output_info = {};
          ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_outputs[0], output_info));
          output_info.shape[output_info.shape.size()-1] = hints.split_size;
          output_info.shape[output_info.shape.size()-2] = hints.act_tile_size;

          if (hints.crouton)
        {
          // With crouton: activations are (8, act_tile_size/8, K), weights are (8, split_size/8, K)
          // After matmul: (8, act_tile_size/8, K) × (K, split_size/8) = (8, act_tile_size/8, split_size/8)
          output_info.shape[output_info.shape.size()-3] = 8;
          output_info.shape[output_info.shape.size()-2] = hints.act_tile_size / 8;
          output_info.shape[output_info.shape.size()-1] = hints.split_size / 8;
          LOGS(logger, INFO) << "Crouton tiling applied to output tensor split, shape: " << output_info.shape[0] << "," << output_info.shape[1] << "," << output_info.shape[2];
        }

          // make some output tensors.
          std::string output_name = node_unit.Name() + "Output_split_" + std::to_string(i) + "_tile_" + std::to_string(j);
          split_output_tensor_names.push_back(output_name);
          LOGS(logger, INFO) << "Added output tensor: " << output_name << " with shape_size " << output_info.shape.size();
          for (size_t k = 0; k < output_info.shape.size(); ++k) {
            LOGS(logger, INFO) << "Output tensor shape[" << k << "]: " << output_info.shape[k];
          }
          QnnTensorWrapper output_tensor_split(
              output_name,
              QNN_TENSOR_TYPE_NATIVE,
              output_info.qnn_data_type,
              std::move(output_info.quant_param),  // If unquantized, otherwise pass scale/offset
              std::move(output_info.shape));
          // add the tensor to the model wrapper.
          ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_split)), "Failed to add output tensor");


        }

        // since we have a double concat, these are the intermediary concat outputs of each set of inner for loops above
        TensorInfo output_info = {};
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_outputs[0], output_info));
        output_info.shape[output_info.shape.size()-1] = hints.split_size;
        output_info.shape[output_info.shape.size()-2] = hints.act_tile_size*hints.act_tile_count;
        if (hints.crouton)
        {
          output_info.shape[output_info.shape.size()-3] = 8;
          output_info.shape[output_info.shape.size()-2] = (hints.act_tile_size * hints.act_tile_count) / 8;
          output_info.shape[output_info.shape.size()-1] = hints.split_size / 8;
          // LOGS(logger, INFO) << "Crouton tiling applied to A tensor split, shape: " << output_info.shape[0] << "," << output_info.shape[1] << "," << output_info.shape[2];
        }
        std::string output_name = node_unit.Name() + "Output_concat_" + std::to_string(i);
        intermediate_concat_output_names.push_back(output_name);

        QnnTensorWrapper output_tensor_split(
              output_name,
              QNN_TENSOR_TYPE_NATIVE,
              output_info.qnn_data_type,
              std::move(output_info.quant_param),  // If unquantized, otherwise pass scale/offset
              std::move(output_info.shape));
          // add the tensor to the model wrapper.
          ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_split)), "Failed to add output tensor");
      }
    }
  } else if (hints.act_tile_count > 1) {
    // if split_count is 1, but act_tile_count > 1, we still need to create the intermediate output tensors.
    for (size_t i = 0; i < hints.act_tile_count; ++i) {
        TensorInfo output_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_outputs[0], output_info));
      output_info.shape[output_info.shape.size()-2] = hints.act_tile_size;

      if (hints.crouton)
      {
        output_info.shape[1] = 8;
        output_info.shape[2] = output_info.shape[2] / 8;
        LOGS(logger, INFO) << "Crouton tiling applied to A tensor split, shape: " << output_info.shape[0] << "," << output_info.shape[1] << "," << output_info.shape[2] << "," << output_info.shape[3];
      }

      // make some output tensors.
      std::string output_name = node_unit.Name() + "Output_tile_" + std::to_string(i);
      split_output_tensor_names.push_back(output_name);
      LOGS(logger, INFO) << "Added output tensor: " << output_name << " with shape_size " << output_info.shape.size() << " act_tile_size: " << hints.act_tile_size;
      for (size_t j = 0; j < output_info.shape.size(); ++j) {
        LOGS(logger, INFO) << "Output tensor shape[" << j << "]: " << output_info.shape[j];
      }
      QnnTensorWrapper output_tensor_split(
          output_name,
          QNN_TENSOR_TYPE_NATIVE,
          output_info.qnn_data_type,
          std::move(output_info.quant_param),  // If unquantized, otherwise pass scale/offset
          std::move(output_info.shape));
      // add the tensor to the model wrapper.
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_split)), "Failed to add output tensor");


    }
  } else {
    // if split_count is 1, we just use the original output tensor.
    split_output_tensor_names.push_back(node_outputs[0].node_arg.Name());
  }

  // regenerate the names that were made in ProcessInputs, due to splitting it is more organized to do it here than to use the passed argument.
  std::vector<std::string> split_b_tensor_names;
  std::vector<std::string> split_scales_tensor_names;
  std::vector<std::string> split_zeros_tensor_names;
  std::vector<std::string> split_a_tensor_names;
  for (size_t i = 0; i < hints.split_count; ++i) {
    split_b_tensor_names.push_back(node_unit.Name() + "B_" + std::to_string(i));
    split_scales_tensor_names.push_back(node_unit.Name() + "Scale_" + std::to_string(i));
    split_zeros_tensor_names.push_back(node_unit.Name() + "Zeros_" + std::to_string(i));
  }

  if (num_tokens == 1) {
    LOGS(logger, INFO) << "Using the MatMulNBits kernel" << do_op_validation;

    for (size_t i = 0; i < hints.split_count; ++i) {
      std::vector<std::string> param_tensor_names = load_parmams_to_qnn(qnn_model_wrapper, node_unit.Index(), kernel_params, node_unit.Name() + "_split_" + std::to_string(i));

      if (hints.scratch) {
        // scratch buffer sizes, maybe move inside a class
        int32_t GROUP_SIZE = 4;
        int32_t LUT_WIDTH = 2 << (GROUP_SIZE - 1);

        size_t lut_size = (kernel_params.K.uint32Value / GROUP_SIZE) * LUT_WIDTH * sizeof(uint16_t);

        size_t scratch_size = lut_size;

        // scratch shape
        std::vector<uint32_t> scratch_shape = {1, 1, 1, (uint32_t)scratch_size};  // This is a placeholder, actual shape will be determined by the kernel.
        std::string scratch_name = node_unit.Name() + "Scratch_" + std::to_string(i);
        QnnTensorWrapper scratch_tensor_wrapper(
            scratch_name,
            QNN_TENSOR_TYPE_NATIVE,
            QNN_DATATYPE_UINT_8,
            std::move(QnnQuantParamsWrapper()),  // If unquantized, otherwise pass scale/offset
            std::move(scratch_shape));
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scratch_tensor_wrapper)), "Failed to add scratch tensor");

        ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_unit.Name() + "_split_" + std::to_string(i),
                                                          "MatMulNBits",
                                                          "MatMulNBits",
                                                          {node_inputs[0].node_arg.Name(), split_b_tensor_names[i], split_scales_tensor_names[i], split_zeros_tensor_names[i]},
                                                          {split_output_tensor_names[i], scratch_name},
                                                          std::move(param_tensor_names),
                                                          do_op_validation),
                          "Failed to add fused MatMulNBits fused node.");

      } else {  // hints.scratch = false
        LOGS(logger, INFO) << "Using the MatMulNBits kernel without scratch buffer";
        ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_unit.Name() + "_split_" + std::to_string(i),
                                                          "MatMulNBits",
                                                          "MatMulNBits",
                                                          {node_inputs[0].node_arg.Name(), split_b_tensor_names[i], split_scales_tensor_names[i], split_zeros_tensor_names[i]},
                                                          {split_output_tensor_names[i]},
                                                          std::move(param_tensor_names),
                                                          do_op_validation),
                          "Failed to add fused MatMulNBits fused node without scratch buffer.");
      }
    }

    if (hints.split_count != 1) {
      std::vector<std::string> param_tensor_names_concat;
      int output_ndim = node_outputs[0].node_arg.Shape()->dim_size();
      int32_t default_axis = output_ndim - 1;
      Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
      axis_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
      axis_qnn_scalar.int32Value = default_axis;
      QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_CONCAT_PARAM_AXIS, axis_qnn_scalar);
      param_tensor_names_concat.push_back(axis_param.GetParamTensorName());
      qnn_model_wrapper.AddParamWrapper(std::move(axis_param));
      // if we are splitting the output, we need to concatenate the outputs.
      std::string concat_name = node_unit.Name() + "Concat";
      LOGS(logger, INFO) << "Creating Concat node: " << concat_name;
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(concat_name,
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        QNN_OP_CONCAT,
                                                        std::move(split_output_tensor_names),
                                                        {node_outputs[0].node_arg.Name()},
                                                        std::move(param_tensor_names_concat),
                                                        do_op_validation),
                        "Failed to add Concat node.");
    }

  } else {  // num_tokens > 1
    LOGS(logger, INFO) << "Using the unpack_weights kernel with regular matmul, num tokens:" << num_tokens;
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

    float scale_scale = 1.0f;
    int32_t scale_zero = 0;
    get_scale_quant_params(qnn_model_wrapper, node_inputs[2], scale_scale, scale_zero, logger);

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

    // If we tile along activations, we need to concat the outputs along two dimensions
    // collect the intermediary output names in this case
    // If this isn't happening, then we just default
    std::vector<std::string> tiling_output_names = split_output_tensor_names;
    if (hints.act_tile_count > 1)
    {
      tiling_output_names.clear();
    }

    // now we make for loop for each of the split weights, scales and zeros tensors.
    for (size_t i = 0; i < hints.split_count; ++i) {
      LOGS(logger, INFO) << "Creating UnpackWeightsNBits node for split: " << i;
      // create the weights tensor name
      std::string weights_name = node_unit.Name() + "_weights_" + std::to_string(i);

      // Unpack weights now transposes within it:
      std::vector<uint32_t> weights_shape = {hints.split_size, kernel_params.K.uint32Value};
      if (hints.kernel_transpose)
      {
        // We need a transposed shape
        weights_shape = {kernel_params.K.uint32Value, hints.split_size};
      }

      if (hints.crouton)
      {
        // With crouton layout, reshape weights to match activation's batch dimension
        // This enables efficient batched matmul where each of the 8 batches multiplies independently
        weights_shape = {8, hints.split_size / 8, kernel_params.K.uint32Value};
        LOGS(logger, INFO) << "Crouton tiling applied to weights tensor, shape: " << weights_shape[0] << "," << weights_shape[1] << "," << weights_shape[2];
      }

      LOGS(logger, INFO) << "Weights " << weights_name << " tensor shape for unpacked weights: " << weights_shape[0] << "," << weights_shape[1];

      QnnTensorWrapper weights_tensor(weights_name,
                                      QNN_TENSOR_TYPE_NATIVE,
                                      QNN_DATATYPE_UFIXED_POINT_8,
                                      std::move(QnnQuantParamsWrapper(scale, offset)),
                                      std::move(weights_shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weights_tensor)), "Failed to add tensor.");

      std::vector<std::string> param_tensor_names_split = load_parmams_to_qnn(qnn_model_wrapper, node_unit.Index(), kernel_params, node_unit.Name() + "_split_" + std::to_string(i));
      std::string unpack_name = node_unit.Name() + "_unpack_" + std::to_string(i);
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(unpack_name,
                                                        "UnpackWeightsNBits",
                                                        "UnpackWeightsNBits",
                                                        {split_b_tensor_names[i], split_scales_tensor_names[i], split_zeros_tensor_names[i]},
                                                        {weights_name},
                                                        std::move(param_tensor_names_split),
                                                        do_op_validation),
                        "Failed to add fused MatMulNBits fused node.");

      std::vector<std::string> param_tensor_names_mul;

      LOGS(logger, INFO) << "Consider tiling A tensors: " << hints.act_tile_count;

      if (hints.act_tile_count > 1 && num_tokens > 1) {
        LOGS(logger, INFO) << "Try to tile A by " << hints.act_tile_count;

        // split the A input properly?
        // and the output of the unpack? This doesn't need to be split yet
        std::vector<std::string> a_tile_names;
        for (size_t j = 0; j < hints.act_tile_count; ++j) {
          std::string split_output_name =
              node_unit.Name() + "Output_" + std::to_string(i) + "_split_" + std::to_string(j);

          TensorInfo input_info = {};
          ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_inputs[0], input_info));

          const size_t last = input_info.shape.size() - 2;

          input_info.shape[last] = hints.act_tile_size;

          // WHy is this failing?
          LOGS(logger, INFO) << "shape size before crouton check: " << input_info.shape.size();
          LOGS(logger, INFO) << "node_inptus[0] shape " << input_info.shape[0] << "," << input_info.shape[1] << "," << input_info.shape[2];;

          if (hints.crouton)
          {

            // add in RESHAPE node to go from 1x1024x1024 to 8x128x1024
            // and then create the split outputs
            input_info.shape[input_info.shape.size() - 3] = 8;
            input_info.shape[input_info.shape.size() - 2] = input_info.shape[input_info.shape.size() - 2] / 8;
            LOGS(logger, INFO) << "Crouton tiling applied to A tensor split, shape: " << input_info.shape[0] << "," << input_info.shape[1] << "," << input_info.shape[2] << "," << input_info.shape[3];
          }

          QnnTensorWrapper split_output_tensor(
              split_output_name,
              QNN_TENSOR_TYPE_NATIVE,
              input_info.qnn_data_type,
              input_info.quant_param.Copy(),
              std::move(input_info.shape));

          ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(split_output_tensor)),
                            "Failed to add split output tensor");
                            // shape is zero here cos we did a std::move, this is actually ok
          LOGS(logger, INFO) << "Creating A tile tensor " << split_output_name  << " with shape size " << input_info.shape.size();
          a_tile_names.push_back(split_output_name);
          LOGS(logger, INFO) << "A tile tensor: " << a_tile_names[j];
          // group_of_output_names.push_back(split_output_name);
        }

        // ----- Split params -----
        std::vector<std::string> split_param_names;

        // axis = last dim
        int output_ndim = node_inputs[0].node_arg.Shape()->dim_size();
        Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
        axis_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
        axis_qnn_scalar.int32Value = output_ndim - 2;

        QnnParamWrapper axis_param(
            node_unit.Index(), node_unit.Name() + "_split_axis_" + std::to_string(i),
            QNN_OP_SPLIT_PARAM_AXIS, axis_qnn_scalar);
        split_param_names.push_back(axis_param.GetParamTensorName());
        qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

        // split_index = cumulative boundaries excluding 0
        // e.g., for 1024 -> 4x256: {256, 512, 768}
        // If crouton layout is used, dimension was reshaped and reduced by factor of 8
        std::vector<uint32_t> split_index;
        int chunk = hints.crouton ? (hints.act_tile_size / 8) : hints.act_tile_size;
        split_index.reserve(hints.act_tile_count ? (hints.act_tile_count - 1) : 0);
        for (uint32_t k = 1; k < hints.act_tile_count; ++k) {
          split_index.push_back(k * chunk);
        }

        // The SPLIT_INDEX tensor is a 1-D param with length = split_index.size()
        std::vector<uint32_t> split_dim{static_cast<uint32_t>(split_index.size())};
        QnnParamWrapper split_param(
            node_unit.Index(), node_unit.Name() + "_split_idx_" + std::to_string(i),
            QNN_OP_SPLIT_PARAM_SPLIT_INDEX,
            std::move(split_dim),
            std::move(split_index));
        split_param_names.push_back(split_param.GetParamTensorName());
        qnn_model_wrapper.AddParamWrapper(std::move(split_param));

        auto AinputName = node_inputs[0].node_arg.Name();
        if (hints.crouton)
        {
          // create reshape node
          // QnnNodeWrapper reshape_node(
          //     node_unit.Index(), node_unit.Name() + "_A_reshape_" + std::to_string(i),
          //     QNN_OP_RESHAPE,
          //     {AinputName},
          //     {AinputName + "_reshaped" + std::t-_string(i)});
          // ORT_RETURN_IF_NOT(qnn_model_wrapper.AddNodeWrapper(std::move(reshape_node)),
          //                   "Failed to add Reshape node.");

          LOGS(logger, INFO) << "Creating Reshape node for A tensor before split for Crouton layout, reshaping " << AinputName;

          auto newOutputName = AinputName + "_reshaped_" + std::to_string(i);

          LOGS(logger, INFO) << "Got new output name: " << newOutputName;

          TensorInfo input_info = {};
          ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_inputs[0], input_info));
          auto oldShape = input_info.shape;

          LOGS(logger, INFO) << "Got old input shape size: " << input_info.shape.size();


          input_info.shape[input_info.shape.size() - 2] = input_info.shape[input_info.shape.size() - 2]/8;
          input_info.shape[input_info.shape.size() - 3] = 8;

          LOGS(logger, INFO) << "adjusted shape to now: " << input_info.shape[0] << ", " << input_info.shape[1] << ", " << input_info.shape[2];

          // SO we cannot create this tensor?
          QnnTensorWrapper reshape_output_tensor(
              newOutputName,
              QNN_TENSOR_TYPE_NATIVE,
              input_info.qnn_data_type,
              input_info.quant_param.Copy(),
              std::move(input_info.shape));

          ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_output_tensor)),
                            "Failed to add reshape output tensor");

          LOGS(logger, INFO) << "Created reshape output tensor wrapper";

          // because we std::moved the damn array
          // LOGS(logger, INFO) << "Reshape from " << oldShape[0] << ", " << oldShape[1] << ", " << oldShape[2] << " to " << input_info.shape[0] << ", " << input_info.shape[1] << ", " << input_info.shape[2];

          ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                              node_unit.Name() + "_A_reshape_" + std::to_string(i),
                              QNN_OP_PACKAGE_NAME_QTI_AISW,
                              QNN_OP_RESHAPE,
                              {AinputName},
                              {newOutputName},
                              {},
                              do_op_validation),
                          "Failed to add resahape node.");

          AinputName = newOutputName;
          LOGS(logger, INFO) << "Created reshape node";

          // is it failing on the creation of the reshape node because the input name tensor doesn't exist?
          // or the next node?
        }

        // Create Split node
        LOGS(logger, INFO) << "Adding split node";
        std::string split_name = node_unit.Name() + "Split_" + std::to_string(i);
        // Make a copy of a_tile_names since CreateQnnNode takes ownership via std::move
        std::vector<std::string> a_tile_names_copy = a_tile_names;
        ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                              split_name,
                              QNN_OP_PACKAGE_NAME_QTI_AISW,
                              QNN_OP_SPLIT,
                              {AinputName},    // input - A
                              std::move(a_tile_names_copy),  // Move the copy, keep original
                              std::move(split_param_names),
                              do_op_validation),
                          "Failed to add Split node.");


        LOGS(logger, INFO) << "SPLIT NODE ADDED /-------------------------------------------";
        LOGS(logger, INFO) << "a_tile_names size before loop: " << a_tile_names.size();
        std::vector<std::string> matmul_output_names;
        for (size_t j = 0; j < hints.act_tile_count; ++j) {

          assert(hints.act_tile_size * hints.act_tile_count == num_tokens);
          LOGS(logger, INFO) << "Activation tile size: " << hints.act_tile_size << ", tile count: " << hints.act_tile_count << ", num tokens: " << num_tokens;


          // Now create the matmul node for this tensor
          // Clear param_tensor_names_mul for each tile iteration to avoid duplicates
          param_tensor_names_mul.clear();

          Qnn_Scalar_t t0 = QNN_SCALAR_INIT;
          t0.dataType = QNN_DATATYPE_BOOL_8;
          t0.bool8Value = 0;
          QnnParamWrapper p0(node_unit.Index(), node_unit.Name() + std::to_string(i) + std::to_string(j), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, t0);
          param_tensor_names_mul.push_back(p0.GetParamTensorName());
          qnn_model_wrapper.AddParamWrapper(std::move(p0));

          Qnn_Scalar_t t1 = QNN_SCALAR_INIT;
          t1.dataType = QNN_DATATYPE_BOOL_8;
          t1.bool8Value = 0;  // transpose the wieght input.
          QnnParamWrapper p1(node_unit.Index(), node_unit.Name() + std::to_string(i) + std::to_string(j), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, t1);
          param_tensor_names_mul.push_back(p1.GetParamTensorName());
          qnn_model_wrapper.AddParamWrapper(std::move(p1));
          std::string matmul_op_name = node_unit.Name() + "_mat_mul_" + std::to_string(i) + "_+_" + std::to_string(j);

          // a_tile_names[j] cannot be accessed. Why?
          LOGS(logger, INFO) << "Created name var " << a_tile_names[j];
          LOGS(logger, INFO) << "Creating MatMul node: " << matmul_op_name << " with tensors " << a_tile_names[j] << " and " << weights_name << " outputting to " << split_output_tensor_names[j];
          ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(matmul_op_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                            QNN_OP_MAT_MUL,
                                                            {a_tile_names[j], weights_name}, {split_output_tensor_names[i*hints.act_tile_count + j]},
                                                            std::move(param_tensor_names_mul), do_op_validation),
                            "Failed to add fused Matmul node.");
          LOGS(logger, INFO) << "Created matmul";
          matmul_output_names.push_back(split_output_tensor_names[i*hints.act_tile_count + j]);
        }


        // Need to concat all the matmul outputs here
        // concat on height axis
        LOGS(logger, INFO) << "Concatenating the outputs of the MatMul nodes on height.";
        // now we need to add the output node, which is a concat of all the matmul outputs.
        std::vector<std::string> param_tensor_names_concat;
        output_ndim = node_outputs[0].node_arg.Shape()->dim_size();
        int32_t default_axis = 1;//output_ndim - 1;
        LOGS(logger, INFO) << "Concatenating on axis: " << default_axis << " of output ndim: " << output_ndim;
        axis_qnn_scalar = QNN_SCALAR_INIT;
        axis_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
        axis_qnn_scalar.int32Value = default_axis;

        QnnParamWrapper concat_axis_param(node_unit.Index(), node_unit.Name() + "_inner_concat_" + std::to_string(i), QNN_OP_CONCAT_PARAM_AXIS, axis_qnn_scalar);
        param_tensor_names_concat.push_back(concat_axis_param.GetParamTensorName());
        qnn_model_wrapper.AddParamWrapper(std::move(concat_axis_param));
        LOGS(logger, INFO) << "Creating Concat node on the output: " << node_unit.Name() + "Concat_inner_" + std::to_string(i);
        ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_unit.Name() + "_concat_inner_" + std::to_string(i),
                                                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                          QNN_OP_CONCAT,
                                                          std::move(matmul_output_names),
                                                          {intermediate_concat_output_names[i]},
                                                          std::move(param_tensor_names_concat),
                                                          do_op_validation),
                          "Failed to add fused Concat node.");
      }
      else
      {
         Qnn_Scalar_t t0 = QNN_SCALAR_INIT;
        t0.dataType = QNN_DATATYPE_BOOL_8;
        t0.bool8Value = 0;
        QnnParamWrapper p0(node_unit.Index(), node_unit.Name() + std::to_string(i), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, t0);
        param_tensor_names_mul.push_back(p0.GetParamTensorName());
        qnn_model_wrapper.AddParamWrapper(std::move(p0));

        Qnn_Scalar_t t1 = QNN_SCALAR_INIT;
        t1.dataType = QNN_DATATYPE_BOOL_8;
        t1.bool8Value = 1;  // transpose the weight input since weights are (N, K) layout
        if (hints.kernel_transpose)
        {
          // transpose already occurs inside the kernel, so weights are already (K, N)
          t1.bool8Value = 0;
        }

        QnnParamWrapper p1(node_unit.Index(), node_unit.Name() + std::to_string(i), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, t1);
        param_tensor_names_mul.push_back(p1.GetParamTensorName());
        qnn_model_wrapper.AddParamWrapper(std::move(p1));
        std::string matmul_op_name = node_unit.Name() + "_mat_mul_" + std::to_string(i);
        LOGS(logger, INFO) << "Creating MatMul node: " << matmul_op_name;
        ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(matmul_op_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                          QNN_OP_MAT_MUL,
                                                          {node_inputs[0].node_arg.Name(), weights_name}, {split_output_tensor_names[i]},
                                                          std::move(param_tensor_names_mul), do_op_validation),
                          "Failed to add fused Matmul node.");
      }
    }

    if (hints.split_count > 1 && hints.act_tile_count > 1) {
      LOGS(logger, INFO) << "Concatenating the outputs of the MatMul nodes.";
      // now we need to add the output node, which is a concat of all the matmul outputs.
      std::vector<std::string> param_tensor_names_concat;
      int output_ndim = node_outputs[0].node_arg.Shape()->dim_size();
      int32_t default_axis = output_ndim - 1;
      LOGS(logger, INFO) << "Concatenating on axis: " << default_axis << " of output ndim: " << output_ndim;
      Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
      axis_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
      axis_qnn_scalar.int32Value = default_axis;
      QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name() + "_outer_concat", QNN_OP_CONCAT_PARAM_AXIS, axis_qnn_scalar);
      param_tensor_names_concat.push_back(axis_param.GetParamTensorName());
      qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

      if (hints.crouton) {
        // For crouton layout: concat to intermediate shape (8, height/8, total_width/8), then reshape to final output
        LOGS(logger, INFO) << "Creating intermediate Concat node for crouton layout";

        // Create intermediate concat output tensor with crouton shape
        TensorInfo concat_output_info = {};
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_outputs[0], concat_output_info));
        concat_output_info.shape[concat_output_info.shape.size()-3] = 8;
        concat_output_info.shape[concat_output_info.shape.size()-2] = (hints.act_tile_size * hints.act_tile_count) / 8;
        concat_output_info.shape[concat_output_info.shape.size()-1] = (hints.split_size * hints.split_count) / 8;

        std::string concat_intermediate_name = node_unit.Name() + "_concat_intermediate";
        QnnTensorWrapper concat_output_tensor(
            concat_intermediate_name,
            QNN_TENSOR_TYPE_NATIVE,
            concat_output_info.qnn_data_type,
            concat_output_info.quant_param.Copy(),
            std::move(concat_output_info.shape));

        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(concat_output_tensor)),
                          "Failed to add intermediate concat output tensor");

        LOGS(logger, INFO) << "Creating Concat node on the output: " << node_unit.Name() + "_concat_outer";
        ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_unit.Name() + "_concat_outer",
                                                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                          QNN_OP_CONCAT,
                                                          std::move(intermediate_concat_output_names),
                                                          {concat_intermediate_name},
                                                          std::move(param_tensor_names_concat),
                                                          do_op_validation),
                          "Failed to add intermediate Concat node.");

        // Now reshape back to original output shape (1, height, width)
        LOGS(logger, INFO) << "Creating final Reshape node to restore original output shape";
        ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                            node_unit.Name() + "_final_reshape",
                            QNN_OP_PACKAGE_NAME_QTI_AISW,
                            QNN_OP_RESHAPE,
                            {concat_intermediate_name},
                            {node_outputs[0].node_arg.Name()},
                            {},
                            do_op_validation),
                        "Failed to add final reshape node.");
      } else {
        // Non-crouton path: concat directly to output
        LOGS(logger, INFO) << "Creating Concat node on the output: " << node_unit.Name() + "_concat_outer";
        ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_unit.Name() + "_concat_outer",
                                                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                          QNN_OP_CONCAT,
                                                          std::move(intermediate_concat_output_names),
                                                          {node_outputs[0].node_arg.Name()},
                                                          std::move(param_tensor_names_concat),
                                                          do_op_validation),
                          "Failed to add final Concat node.");
      }
    } else if (hints.split_count > 1)
    {
      LOGS(logger, INFO) << "Concatenating the outputs of the MatMul nodes.";
      // now we need to add the output node, which is a concat of all the matmul outputs.
      std::vector<std::string> param_tensor_names_concat;
      int output_ndim = node_outputs[0].node_arg.Shape()->dim_size();
      int32_t default_axis = output_ndim - 1;
      LOGS(logger, INFO) << "Concatenating on axis: " << default_axis << " of output ndim: " << output_ndim;
      Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
      axis_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
      axis_qnn_scalar.int32Value = default_axis;
      QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_CONCAT_PARAM_AXIS, axis_qnn_scalar);
      param_tensor_names_concat.push_back(axis_param.GetParamTensorName());
      qnn_model_wrapper.AddParamWrapper(std::move(axis_param));
      LOGS(logger, INFO) << "Creating Concat node on the output: " << node_unit.Name() + "Concat";
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_unit.Name() + "_concat",
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        QNN_OP_CONCAT,
                                                        std::move(split_output_tensor_names),
                                                        {node_outputs[0].node_arg.Name()},
                                                        std::move(param_tensor_names_concat),
                                                        do_op_validation),
                        "Failed to add fused Concat node.");
    }
  }

  return Status::OK();
}
}  // namespace qnn
}  // namespace onnxruntime
